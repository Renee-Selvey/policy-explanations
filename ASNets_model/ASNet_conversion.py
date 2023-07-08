import re
import gurobipy as gp
from gurobipy import GRB

from asnets.prob_dom_meta import BoundAction, BoundProp
from asnets.state_reprs import get_init_cstate
from asnets.scripts.weights2equations import snapshot_to_equations

def create_layers(m):
    '''
    Create the variables and constraints for the inner layers of the ASNet
    pre-activation
    '''
    weights = m.network.instantiator.snapshot_path

    # Find the total number of layers in the network
    num_layers = len(m.network.instantiator.weight_manager.prop_weights) \
            + len(m.network.instantiator.weight_manager.act_weights)

    # Get sparse equations from asnets script
    equations = snapshot_to_equations(weights)

    # Get bound actions and propositions and convert to dict form
    bound_props = convert_bound_to_dict(
        m.network.planner_exts.problem_meta.bound_props_ordered
    )
    bound_actions = convert_bound_to_dict(
        m.network.planner_exts.problem_meta.bound_acts_ordered
    )

    # Create variables for the network pre-activation
    elu_in, elu_out, connected_modules = create_vars(
        m,
        equations,
        num_layers,
        bound_props,
        bound_actions
    )

    # Add the constraints for the network pre-activation
    create_constrs(
        m,
        equations,
        num_layers,
        bound_actions,
        bound_props,
        elu_in,
        connected_modules
    )

    # Return the input variables and modules dictionaries
    return elu_in, elu_out

def convert_bound_to_dict(bound_tuple):
    '''
    Converts the tuple of bound actions (propositions) to dictionaries where
    the keys are the lifted action (proposition) name and the values are a set 
    of all grounded versions of it.
    '''
    bound_dict = {}
    for bound_obj in bound_tuple:
        if isinstance(bound_obj, BoundProp):
            name = bound_obj.pred_name
        elif isinstance(bound_obj, BoundAction):
            name = bound_obj.prototype.schema_name

        if name in bound_dict.keys():
            bound_dict[name].add(bound_obj)
        else:
            bound_dict[name] = {bound_obj}

    return bound_dict

def create_vars(
    m,
    equations,
    num_layers,
    bound_props,
    bound_actions,
):
    '''
    Creates the variables for the inner layers of an ASNet
    '''
    # keys := elu_in/elu_out/out[bound_act/prop_[layer,index]]
    # values := gurobi variables
    elu_in = {}  
    elu_out = {}
    # keys := bound_act/prop_[layer,index]
    # values := 
    #       for proposition layers:
    #           a list, for pooled variables, a variable that is the max of all
    #           related propositions in that slot and for skip connections a 
    #           single related variable
    #       for action layers:
    #           a list of related proposition variables 
    connected_modules = {}

    # Get the ids of the heuristic data inputs
    heur_types = get_init_cstate(m.network.planner_exts)._aux_dat_interp
    # Dictionary to store the LM-cut heuristics for each applicable bound 
    # actions, as only one may be true (They are mutually exclusive)
    LM_cut = {}

    for global_layer_num, layer in enumerate(equations):
        # Action layer if it is an even layer else it is a proposition layer
        bound_dict = bound_actions \
            if global_layer_num % 2 == 0 else bound_props
        
        for eqn in layer:
            layer_num = eqn.out_la.layer
            index = eqn.out_la.index
            
            for act_or_prop in bound_dict[eqn.out_la.act_or_prop_id]:
                # Create a Gurobi variable for each outgoing 'lifted activation' 
                # for each equation
                name = f"{underscore_unique_ident(act_or_prop)}[{layer_num},{index}]"
                final_layer = num_layers // 2
                
                # Don't need to compute elu activation for final layer
                if layer_num == final_layer:
                    m._out[name] = m.model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=-GRB.INFINITY,
                        name=f"out[{name}]"
                    )
                else:
                    elu_in[name] = m.model.addVar(
                        vtype=GRB.CONTINUOUS, 
                        lb=-GRB.INFINITY, 
                        name=f"elu_in[{name}]"
                    )
                    elu_out[name] = m.model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=-1,
                        name=f"elu_out[{name}]"
                    )

                # If it is an action layer, we need to connect lifted parameters
                # to the problem's objects
                if isinstance(act_or_prop, BoundAction):
                    lifted_params = eqn.out_la.backing_obj.param_names
                    params_to_objects = {}
                    for i, param in enumerate(lifted_params):
                        params_to_objects[param] = act_or_prop.arguments[i]

                # Connect all incoming variables of the equation to the outgoing
                # variable in the connected_modules dictionary
                connected = []
                for conn_name in eqn.in_la_list:
                    conn_schema, conn_layer, conn_index = conn_name.ident()
                    var = None
                    
                    # Add the skip connection from the previous layer
                    grounded_type = BoundAction if conn_name.role == "act" else BoundProp
                    if isinstance(act_or_prop, grounded_type): # must be the skip connection
                        conn_id = f"{underscore_unique_ident(act_or_prop)}[{conn_layer},{conn_index}]"
                        try:
                            var = elu_in[conn_id]
                        except KeyError:
                            # raise KeyError(f"can't find {conn_id}")
                            # The skip module doesn't exist because it has been zeroed out
                            var = 0

                    elif isinstance(act_or_prop, BoundAction):
                        if layer_num == 0:
                            # Get the schema name of the input
                            find_schema = re.search("\((.+?)\(", conn_name.name)
                            if find_schema:
                                conn_schema = find_schema.group(1)
                            
                            assert conn_schema is not None, \
                                (f"Input name '{conn_name.name}' doesn't match "
                                "the expected format, so schema name can't be "
                                "extracted")

                            # Get the lifted variables of the input
                            lifted_vars = re.findall(
                                "\?v[0-9]+", conn_name.name
                            )

                            conn_id = f"{conn_schema}"
                            for lifted_var in lifted_vars:
                                conn_id += f"_{params_to_objects[lifted_var]}"

                            # Create Gurobi variable for each input variable
                            conn_id = re.sub("\(.*\)", f"({conn_id})", conn_name.name)
                            # Don't want to add the same variable twice
                            if conn_id not in m._network_input:
                                if conn_id.startswith("heuristic_data"):
                                    # Integer if the the feature is
                                    # action count
                                    aux_data_ind = int(
                                        re.search("\[([0-9]+)\]", conn_id).group(1)
                                    )
                                    heur_id = heur_types[aux_data_ind]
                                    if heur_id == "action_count":
                                        m._network_input[conn_id] = \
                                            m.model.addVar(
                                                vtype=GRB.INTEGER,
                                                name=conn_id
                                            )

                                        m.input_vars[conn_id] = [
                                            m._network_input[conn_id],
                                            False
                                        ]
                                    # Binary if the feature is is-enabled
                                    elif heur_id == "is-enabled":
                                        add_enabled_var(m, conn_id)
                                    # Binary if the feature is related to LM-cut
                                    # heuristic (is mutually exclusive)
                                    elif heur_id in {
                                        "in-any-cut", 
                                        "in-singleton-cut", 
                                        "in-last-cut",
                                    }:
                                        m._network_input[conn_id] = \
                                            m.model.addVar(
                                                vtype=GRB.BINARY,
                                                name=conn_id
                                            )

                                        m.input_vars[conn_id] = [
                                            m._network_input[conn_id],
                                            False
                                        ]

                                        # Add LM-Cut heuristic to dict
                                        k = re.search("\((.*)\)", conn_id).group(1)
                                        LM_cut.setdefault(
                                            k, []
                                        ).append(m._network_input[conn_id])
                                    else:
                                        raise NotImplementedError(
                                            f'''Can't create a variable for 
                                            {conn_id} because we don't know the 
                                            type of {heur_id}'''
                                        )
                                
                                else: # is_true or is_goal
                                    if conn_id not in m.input_vars.keys():
                                        m._network_input[conn_id] = \
                                            m.model.addVar(
                                                vtype=GRB.BINARY,
                                                name=conn_id
                                            )
                                    
                                        m.input_vars[conn_id] = [
                                            m._network_input[conn_id],
                                            False
                                        ]
                                    else:
                                        m._network_input[conn_id] = \
                                            m.input_vars[conn_id][0]

                            var = m._network_input[conn_id]
                            
                        else:
                            # Get the variable's key
                            conn_id = f"{conn_schema}"
                            
                            # Need to match the objects associated to the parameters 
                            # of the outgoing 'lifted activation' to the same 
                            # parameters of the incoming 'lifted activations'
                            for lifted_var in conn_name.backing_obj.params:
                                conn_id += f"_{params_to_objects[lifted_var]}"
                            conn_id += f"[{conn_layer},{conn_index}]"

                            var = elu_out[conn_id]
                    
                    elif isinstance(act_or_prop, BoundProp):
                        # All non-skip connections in the proposition layer are
                        # pooled
                        pool_name = f"pool_{conn_schema}@{conn_name.slot}" \
                            f"[{conn_layer},{conn_index}]_for_{name}"

                        # Create pool variable
                        var = m.model.addVar(
                            vtype=GRB.CONTINUOUS, 
                            lb=-1, 
                            name=pool_name
                        )

                        pool = []
                        # Find the prop in the related props of the bound 
                        # actions (in the correct position) to add to the 
                        # pooled list
                        for act in bound_actions[conn_name.act_or_prop_id]:
                            if act.props[conn_name.slot].unique_ident == act_or_prop.unique_ident:
                                conn_id = f"{underscore_unique_ident(act)}[{conn_layer},{conn_index}]"
                                pool.append(elu_out[conn_id])

                        
                        if pool:
                            # Add the constraint which max pools the variables
                            m.model.addConstr(var == gp.max_(pool))
                        else:
                            # We can add a constant as -1 for empty pooled objects
                            # as the minimum output of an elu activation is -1
                            # From Sam (regarding a similar issue in tf):
                            #   we get around that problem by clamping the value 
                            #   to the minimum possible activation output from 
                            #   the last layer
                            m.model.addConstr(var == -1)

                    if var:
                        connected.append(var)

                connected_modules[name] = connected

    # Add LM-cut mutually exclusive constraints
    for heurs in LM_cut.values():
        m.model.addConstr(gp.quicksum(heurs) == 1)

    m.model.update()
    return elu_in, elu_out, connected_modules

def underscore_unique_ident(bound_obj):
    '''
    Returns the unique_ident of the input BoundProp/BoundAction with hyphens
    instead of spaces as Gurobi doesn't like spaces in names
    '''
    return bound_obj.unique_ident.replace(" ", "_")

def add_enabled_var(m, conn_id):
    '''
    Adds is-enabled variables and variables for each of it's preconditions.
    
    Also adds constraints to link is-enabled to the precondition vars
    '''
    key = re.search("\((.*)\)", conn_id).group(1)

    # Add variable to the network input
    m._network_input[conn_id] = m.model.addVar(vtype=GRB.BINARY, name=conn_id)

    # If the action is cancelled by FD, it will never be chosen - set as not
    # enabled
    if key not in m.non_trivial_actions:
        m.model.addConstr(m._network_input[conn_id] == 0)
        
        m.model.update()
        return
    
    # Add all of the preconditions to variable to the input variables
    pos_pre = {}
    for pred in m._ground_actions[key].positive_preconditions:
        var_key = f"{pred[0]}"
        for obj in pred[1:]:
            var_key += f"_{obj}"

        var_key = f"is_true({var_key})"

        if var_key not in m.input_vars.keys():
            m.input_vars[var_key] = [
                m.model.addVar(vtype=GRB.BINARY, name=var_key),
                False
            ]

        pos_pre[var_key] = m.input_vars[var_key][0]

        # is-enabled must be less than all the positive preconditions
        m.model.addConstr(m._network_input[conn_id] <= pos_pre[var_key])
    
    neg_pre = {}
    for pred in m._ground_actions[key].negative_preconditions:
        var_key = f"{pred[0]}"
        for obj in pred[1:]:
            var_key += f"_{obj}"

        var_key = f"is_true({var_key})"

        if var_key not in m.input_vars.keys():
            m.input_vars[var_key] = [
                m.model.addVar(vtype=GRB.BINARY, name=var_key),
                False
            ]
        
        neg_pre[var_key] = m.input_vars[var_key][0]

        # is-enabled must be less than the negation of negative preconditions
        m.model.addConstr(m._network_input[conn_id] <= 1 - neg_pre[var_key])

    # is-enabled can only be true when all of the preconditions are satisfied
    pred_conj = [pred for pred in pos_pre.values()]
    pred_conj += [1 - pred for pred in neg_pre.values()]    

    m.model.addConstr(m._network_input[conn_id] >= gp.quicksum(pred_conj) \
        - (len({**pos_pre, **neg_pre}) - 1))

    m.model.update()

def create_constrs(
    m, 
    equations, 
    num_layers, 
    bound_actions, 
    bound_props, 
    elu_in, 
    connected_modules
):
    '''Creates the constraints connecting the inner layers of the ASNet model'''
    # Add the constraints that connect each variable to the related variables in
    # the next layer
    for global_layer_num, layer in enumerate(equations):
        for eqn in layer:
            layer_num = eqn.out_la.layer
            if layer_num == num_layers // 2:
                modules = m._out
            else:
                modules = elu_in
            
            index = eqn.out_la.index

            # Find whether or not it is a action or proposition layer
            bound_dict = bound_actions \
                if global_layer_num % 2 == 0 else bound_props

            for act_or_prop in bound_dict[eqn.out_la.act_or_prop_id]:
                # Get variables/weights/bias
                name = f"{underscore_unique_ident(act_or_prop)}[{layer_num},{index}]"
                mod_out = modules[name]
                mods_in = connected_modules[name]
                weights = eqn.coeffs
                bias = eqn.bias

                # Add constraint
                m.model.addConstr(mod_out == gp.quicksum(
                    w*x for w,x in zip(weights, mods_in)
                ) + bias)
    
    m.model.update()