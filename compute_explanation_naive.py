import argparse
import logging
import re
import tensorflow as tf
from time import process_time, strftime
import gurobipy as gp
from gurobipy import GRB


from asnets.interactive_network import NetworkInstantiator
from ASNets_compute_trajectory import roll_out_trajectory
from ASNets_model.ASNet_conversion import underscore_unique_ident
from ASNets_model.gurobi_network import Model
from sensitivity_calculation import compute_sensitivity

def instantiate_network(
    weights,
    domain,
    problem,
    use_lm_cuts,
    use_history,
    heuristic_data_gen_name,
):
    '''Instantiates the ASNet model for a specific problem'''
    # Need to instantiate the network
    net_instantiator = NetworkInstantiator(
        weights,
        extra_ppddl=[domain],
        use_lm_cuts=use_lm_cuts,
        use_history=use_history,
        heuristic_data_gen_name=heuristic_data_gen_name,
    )
    net_container = net_instantiator.net_for_problem([problem])
    
    return net_container
        
def main_naive(args):
    st = process_time()
    # Create one big MILP for the whole trajectory
    with tf.Session(graph=tf.Graph()) as sess:
        # Instantiate network
        network = instantiate_network(
            args.weights,
            args.domain,
            args.problem,
            use_lm_cuts=args.use_lm_cut,
            use_history=args.use_history,
            heuristic_data_gen_name=None, 
        )

        sess.run(tf.global_variables_initializer())
        states, actions = \
            roll_out_trajectory(network.policy, network.planner_exts)[:2]
    
        domain = args.domain
        
        if not states[-1].is_terminal or len(actions) == 0:
            return
        
        models = []
        for i in range(len(actions)):
            model = Model(network, domain, args.problem, args.use_mutex)
            model.initialise_temp_constraints(states[i])
            model.add_tiebreaking_constraints()
                
            models.append(model)
            
        models[0].model.write("original_model.lp")
            
        big_model = gp.Model("big_network")
        varDict = {}
        # key: act_key, value: [pos_effects, neg_effects]
        effects_dict = {}
        
        # Copy variables from each step's MILP
        for i, m in enumerate(models):
            for v in m.model.getVars():
                new_var = big_model.addVar(
                    lb=v.lb,
                    ub=v.ub,
                    obj=v.obj,
                    vtype=v.vtype,
                    name=f"step_{i}_{v.varname}"
                )
                varDict[f"step_{i}_{v.varname}"] = new_var
        
        # Copy constraints from each step's MILP     
        for j, m in enumerate(models):
            for c in m.model.getConstrs():
                expr = m.model.getRow(c)
                newexpr = gp.LinExpr()
                for i in range(expr.size()):
                    v = expr.getVar(i)
                    coeff = expr.getCoeff(i)
                    newv = varDict[f"step_{j}_{v.Varname}"]
                    newexpr.add(newv, coeff)
                big_model.addConstr(
                    newexpr,
                    c.Sense,
                    c.RHS,
                    name=f"step_{j}_{c.ConstrName}"
                )
                
            for c in m.model.getGenConstrs():
                constr_type = c.getAttr("GenConstrType")
                
                if constr_type == 0: # Max
                    resvar, invars, constant = m.model.getGenConstrMax(c)
                    big_model.addGenConstrMax(
                        varDict[f"step_{j}_{resvar.varname}"],
                        [varDict[f"step_{j}_{v.varname}"] for v in invars],
                        constant
                    )
                elif constr_type == 3: # And
                    resvar, invars = m.model.getGenConstrAnd(c)
                    big_model.addGenConstrAnd(
                        varDict[f"step_{j}_{resvar.varname}"],
                        [varDict[f"step_{j}_{v.varname}"] for v in invars]
                    )
                elif constr_type == 5: # Indicator
                    binvar, binval, expr, sense, rhs = m.model.getGenConstrIndicator(c)
                    newexpr = gp.LinExpr()
                    for i in range(expr.size()):
                        v = expr.getVar(i)
                        coeff = expr.getCoeff(i)
                        newv = varDict[f"step_{j}_{v.Varname}"]
                        newexpr.add(newv, coeff)
                    
                    big_model.addGenConstrIndicator(
                        varDict[f"step_{j}_{binvar.varname}"],
                        binval,
                        newexpr,
                        sense,
                        rhs)
                elif constr_type == 8: # Exponential
                    xvar, yvar = m.model.getGenConstrExp(c)
                    big_model.addGenConstrExp(
                        varDict[f"step_{j}_{xvar.varname}"],
                        varDict[f"step_{j}_{yvar.varname}"]
                    )
                
        # Variables to ensure the next state is the effects of the most recent
        # chosen action
        # Get positive and negative effects of the action
        for act_key in models[0]._max_ind.keys():            
            # Transform the effects into the same syntax
            pos_effects = set()
            neg_effects = set()
            dict_act_key = act_key[:-5]
            for pred in models[0]._ground_actions[dict_act_key].add_effects:
                var_key = f"{pred[0]}"
                for obj in pred[1:]:
                    var_key += f"_{obj}"
                
                var_key = f"is_true({var_key})"
                pos_effects.add(var_key)

            for pred in models[0]._ground_actions[dict_act_key].del_effects:
                var_key = f"{pred[0]}"
                for obj in pred[1:]:
                    var_key += f"_{obj}"
                
                var_key = f"is_true({var_key})"
                neg_effects.add(var_key)
                
            effects_dict[act_key] = [pos_effects, neg_effects]
        
        for i, m in enumerate(models[:-1]):
            act_key = underscore_unique_ident(actions[i]) + "[2,0]"
            pos_effects, neg_effects = effects_dict[act_key]  
                
            # This action being chosen implies the positive effects are true and
            # the negative effects are false
            for pos_effect_keys in pos_effects:
                if f"step_{i+1}_{pos_effect_keys}" in varDict:
                    big_model.addConstr(
                        varDict[f"step_{i+1}_{pos_effect_keys}"] == 1
                    )
                
            for neg_effect_keys in neg_effects:
                if f"step_{i+1}_{neg_effect_keys}" in varDict:
                    big_model.addConstr(
                        varDict[f"step_{i+1}_{neg_effect_keys}"] == 0
                    )
                
            for p in models[0].input_vars:
                if p not in pos_effects and p not in neg_effects:
                    big_model.addConstr(
                        varDict[f"step_{i+1}_{p}"] == varDict[f"step_{i}_{p}"]
                    )
                
        # Goal props stay the same for every step
        for prop, _ in states[0].props_true:
            prop_key = underscore_unique_ident(prop)
            
            if f"step_0_is_goal({prop_key})" in varDict:        
                for i in range(1, len(models)):
                    big_model.addConstr(
                        varDict[f"step_0_is_goal({prop_key})"] \
                            == \
                        varDict[f"step_{i}_is_goal({prop_key})"]
                    )
        
        # For each step of the trajectory, the chosen action must be the one in the
        # trajectory
        big_model.addConstr(
            gp.quicksum(
                varDict[f"step_{i}_max_ind[{underscore_unique_ident(a)}[2,0]]"]
                for i, a in enumerate(actions)
            ) <= len(actions) - 1
        )
        
        # Compute sensitivity formula for each proposition
        model.init_exp = [
            prop
            for prop in model.input_vars
        ]
        sensitivity = compute_sensitivity(model.network, model.init_exp, states, actions)
        removal_order = sorted(
            sensitivity,
            key=sensitivity.get
        )
        removal_order_reverse = [p for p in reversed(removal_order)]
            
    # Start with all variables constrained
    input_var_constraints = []
    vars_in_exp = []
    
    for in_id in removal_order_reverse:
        prop_name = re.search("\((.*)\)", in_id).group(1)
        if in_id.startswith("is_true"):
            constr = big_model.addConstr(varDict[f"step_{0}_{in_id}"] == models[0]._is_true[prop_name])
            input_var_constraints.append(constr)
            vars_in_exp.append([in_id, False])
        elif in_id.startswith("is_goal"):
            big_model.addConstr(varDict[f"step_{0}_{in_id}"] == models[0]._is_goal[prop_name])
        else: # heuristic_data
            aux_data_ind = int(re.search("\[([0-9]+)\]", in_id).group(1))
            constr = big_model.addConstr(
                varDict[f"step_{0}_{in_id}"] == models[0]._heuristic_data[prop_name][aux_data_ind]
            )

    # Now run the algorithm    
    for i, c in enumerate(input_var_constraints):
        if not vars_in_exp[i][0].startswith("is_goal"):
            big_model.update()
            lhs, sense, rhs, name = big_model.getRow(c), c.Sense, c.RHS, c.ConstrName
            big_model.remove(c)
            
            # big_model.setParam(GRB.Param.Presolve, 0)
            big_model.setParam(GRB.Param.FuncPieces, -1)
            big_model.setParam(GRB.Param.FuncPieceLength, 1e-5)
            big_model.setParam(GRB.Param.FuncPieceError, 1e-6)
            big_model.setParam(GRB.Param.IntFeasTol, 1e-9)
            # big_model.setParam(GRB.Param.TimeLimit, 7200)
            big_model.write("big_model.lp")
            big_model.update()
            big_model.optimize()
                        
            if big_model.Status == 2: # can find a solution, not an explanation
                # add constr back in
                big_model.addConstr(lhs, sense, rhs, name)
                vars_in_exp[i][1] = True
                
            big_model.reset()
    

    # Create log file
    timestr = strftime("%Y%m%d-%H%M%S")
    file_name = args.path + "/" + network.single_problem.name + "-" + timestr + ".log"
    logging.basicConfig(filename=file_name, filemode="a", level=logging.INFO)
    
    # Get problem name
    logging.info(f"Problem path: {args.problem}")
    
    # Was the goal reached?
    logging.info(f"Goal reached/Finished in time: {int(states[-1].is_terminal)}")
    
    # Runtime (s)
    # logging.info(f"CPU Runtime: {et_cpu}")
    logging.info(f"Runtime: {process_time() - st}")
    
    # Plan length (number of actions in the trajectory)
    logging.info(f"Plan length: {len(actions)}")
    
    # Size of explanation
    logging.info(f"Size of explanation: {len([v for v in vars_in_exp if v[1] == True])}")
    
    # Size of input
    logging.info(f"Size of input: {len([v for v in vars_in_exp if not v[0].startswith('is_goal')])}")
    
    # Final explanation
    logging.info(f"Explanation: {vars_in_exp}")

parser = argparse.ArgumentParser()
parser.add_argument("weights")
parser.add_argument("domain")
parser.add_argument("problem")
parser.add_argument("--type", required=True, choices=(
    "enum_exp", "plan_exp", "sub_min_exp"
))
parser.add_argument("--path", required=True)
parser.add_argument("--use_lm_cut", action="store_true")
parser.add_argument("--no-use_lm_cut", action="store_false", dest="use_lm_cut")
parser.add_argument("--use_history", action="store_true")
parser.add_argument("--no-use_history", action="store_false", dest="use_history")
parser.add_argument("--use_mutex", action="store_true")
parser.add_argument("--no-use_mutex", action="store_false", dest="use_mutex")
parser.set_defaults(use_lm_cut=False, use_history=False, use_mutex=True)

if __name__ == '__main__':
    main_naive(parser.parse_args())
