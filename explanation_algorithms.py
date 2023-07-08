import re
import copy

from asnets.state_reprs import get_init_cstate

from ASNets_model.ASNet_conversion import underscore_unique_ident

def print_exp(model, explanation):
    '''
    Convert an explanation to an easy to read string format.

    model: Gurobi model for the network
    explanation: a set of the keys to the explanation
    '''
    # Get the ids of the heuristic data inputs
    heur_types = get_init_cstate(model.network.planner_exts)._aux_dat_interp

    # Find the value of each element in the explanation
    exp_values = {}
    for pred in explanation:
        key = re.search("\((.*)\)", pred).group(1)
        if pred.startswith("is_true"):
            exp_values[key] = int(model._is_true[key])
        elif pred.startswith("is_goal"):
            exp_values[pred] = int(model._is_goal[key])
        else: # heuristic data
            # The order (and associated index) of the heuristic data defines
            # what it is
            aux_data_ind = int(re.search("\[([0-9]+)\]", pred).group(1))
            heur_type = heur_types[aux_data_ind]
            
            exp_values[f"{heur_type}({key})"] = \
                int(model._heuristic_data[key][aux_data_ind])

    for k, v in exp_values.items():
        model.exp += f"{k} = {v}\n"
        print(f"\t{k} = {v},")

def subset_min_exp(model, states, actions, order, verbose=True):
    '''
    Find a subset-minimal trajectory explanation for a given trajectory

    model: Gurobi model for the network
    states: list of states in the trajectory
    actions: list of actions in the trajectory
    '''
    # Initialise the explanations for each step of the trajectory - initially
    # contains all literals for each state
    U = []
    for i in range(len(actions)):
        U.append({var for var in model.input_vars if not var.startswith("is_goal")})
        
    # Save the initial explanation to iterate over
    # model.init_exp = [prop for prop in {var for var in model.input_vars}]
    model.init_exp = [
        prop
        for prop in model.input_vars
        if not prop.startswith("is_goal")
    ]
    
    # Also save a deep copy of U so we don't add any elements which have been
    # removed by the mutexes in the iteration
    U_copy = copy.deepcopy(U)
    
    model.sanity_check(states[0], actions[0])

    # Only want to iterate over the literals which haven't already been
    # cancelled out
    num = 0
    for l in order:
        num += 1
        print(num, l)
        
        # Check each the literal can be removed from each step of the
        # explanation
        is_exp = True
        
        for i in range(len(actions)):            
            print(i, actions[i])
            
            # Remove the literal
            U[i].remove(l)
            
            model.initialise_temp_constraints(states[i], actions[i])
            # Initialise the input
            for var in model.input_vars:
                if var in U[i]:
                    model.input_vars[var][1] = True
                elif var.startswith("is_goal"):
                    model.input_vars[var][1] = True
                else:
                    model.input_vars[var][1] = False

            # Check whether it is an explanation
            curr_is_exp = model.is_explanation()
            model.remove_temp_constraints(reset=True)
            
            if not curr_is_exp:
                print("Not explanation, breaking")               
                is_exp = False
                break
            else:
                print("Is explanation")
            
            # If the literal does not appear in the effects of the action
            # we break as there will be no change to the sets of the remaining
            # steps, so they have already been verified as being explanations
            if l.startswith("is_true"):
                action_key = underscore_unique_ident(actions[i])

                effects = set()
                for pos in model._ground_actions[action_key].add_effects:
                    var_key = f"{pos[0]}"
                    for obj in pos[1:]:
                        var_key += f"_{obj}"
                    
                    var_key = f"is_true({var_key})"
                    effects.add(var_key)
                for neg in model._ground_actions[action_key].del_effects:
                    var_key = f"{neg[0]}"
                    for obj in neg[1:]:
                        var_key += f"_{obj}"
                    
                    var_key = f"is_true({var_key})"
                    effects.add(var_key)

                if l in effects:
                    print("I'm breaking")
                    # print("literal", l)
                    # print("action", action_key)
                    break
            
        if not is_exp:
            # Add the literal back in for all previous steps
            for j in range(i, -1, -1):
                if l in U_copy[j]:
                    U[j].add(l)

    if verbose:
        model.initialise_temp_constraints(states[0], actions[0])
        print("Explanation:")
        print_exp(model, U[0])
        model.remove_temp_constraints(reset=True)

    return U[0]