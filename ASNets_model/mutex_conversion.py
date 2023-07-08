import re
import subprocess
import os
import os.path as osp
import gurobipy as gp

from asnets.state_reprs import get_init_cstate

THIS_DIR = osp.dirname(osp.abspath(__file__))
ABOVE_DIR = osp.abspath(osp.join(THIS_DIR, "../../"))
FD_DIR = osp.join(ABOVE_DIR, "downward")
TRANSLATE_PATH = osp.join(FD_DIR, "fast-downward.py")

def run_FD(model):
    '''
    Runs FD translation to get later get mutexes and non-trivial operators
    (actions)
    '''
    # Create the SAS file
    translate = subprocess.Popen(
        [
            "python",
            TRANSLATE_PATH,
            "--translate",
            model.domain_path,
            model.problem_path,
        ]
    )

    translate.wait()
    
    with open("output.sas", "r") as f:
        lines = [l.strip() for l in f.readlines()]
        
        # Get the operators not cancelled out to set the rest as not enabled    
        line_num = lines.index("begin_operator")
        operators_indicies = [
            i
            for i,x in enumerate(lines)
            if x == "begin_operator"
        ]
        
        for ind in operators_indicies:
            action_key = re.sub('\s+', '_', lines[ind+1])
            model.non_trivial_actions.add(action_key)
            
    model.model.update()

def add_mutex_constrs(model):
    with open("output.sas", "r") as f:
        # Will progessively work down the file and extract all useful
        # information from the SAS translation
        lines = [l.strip() for l in f.readlines()]
        line_num = 6
        
        # Get the number of SAS variables in the problem
        num_vars = int(lines[line_num])
        
        # Get the number of values in the first variable
        line_num += 4
        num_val = int(lines[line_num])
        
        # List of lists to store variables and each of their possible values -
        # need a separate list for the variables and goal variables
        SAS_vars = []
        SAS_vars_goal = []
        
        # Convert the SAS variables into the same format as the keys of
        # variables in the gurobi network
        i = 0
        while True:
            var_vals = []
            var_vals_goal = []
            
            for _ in range(num_val):
                line_num += 1
                
                # Add the model variables to their corresponding position in the
                # list if they exist - None otherwise
                var_key = convert_name(lines[line_num])
                if var_key is None:
                    var_vals.append(None)
                elif var_key[0] in model.input_vars:
                    var_vals.append(var_key)
                else:
                    var_vals.append(None)
                    
                var_key_goal = convert_name(lines[line_num], True)
                if var_key is None:
                    var_vals.append(None)
                elif var_key_goal[0] in model.input_vars:
                    var_vals_goal.append(var_key_goal)
                else:
                    var_vals_goal.append(None)
                        
            SAS_vars.append(var_vals)
            SAS_vars_goal.append(var_vals_goal)
            
            # Save the keys of the mutex to the model
            mutex_group = tuple([v for v in var_vals if v is not None])
            mutex_group_goal = tuple([v for v in var_vals_goal if v is not None])

            if mutex_group:
                model.mutexes.add(mutex_group)
            if mutex_group_goal:
                model.mutexes.add(mutex_group_goal)
                    
            i += 1
            if i >= num_vars:
                break
            
            line_num += 5
            num_val = int(lines[line_num])
            
        remove_pruned_props(model, SAS_vars, SAS_vars_goal)
            
        # Make sure that values of each SAS variable are mutex
        # Some of these constraints will be trivial, but that doesn't matter
        for var_keys in [SAS_vars, SAS_vars_goal]:
            for SAS_var in var_keys:
                sum_constr = []
                any_None = False
                for val in SAS_var:
                    if val is None:
                        any_None = True
                    else:
                        SAS_val, not_negated = val
                        if not_negated:
                            sum_constr.append(model.input_vars[SAS_val][0])
                        else:
                            sum_constr.append(1 - model.input_vars[SAS_val][0])
                        
                if any_None:
                    model.model.addConstr(gp.quicksum(sum_constr) <= 1)
                else:
                    model.model.addConstr(gp.quicksum(sum_constr) == 1)
                    
        # Find the mutex conditions in SAS file and add these as constraints
        line_num += 2
        num_mutex = int(lines[line_num])
        
        # Get the number of vars in the current mutex
        line_num += 2
        num_curr_mutex = int(lines[line_num])
        
        i = 0
        while True:
            # Separate lists for the is_true variables and the is_goal variables
            mutex_vars = []
            mutex_vars_goal = []
            
            mutex_group = []
            mutex_group_goal = []

            for _ in range(num_curr_mutex):
                line_num += 1
                
                # Extract the position of the mutexes in the SAS_var list
                var_idx, val_idx = [int(m) for m in lines[line_num].split()]
                
                if SAS_vars[var_idx][val_idx] is not None:
                    mutex_group.append(SAS_vars[var_idx][val_idx])
                    SAS_val, not_negated = SAS_vars[var_idx][val_idx]
                    if not_negated:
                        mutex_vars.append(model.input_vars[SAS_val][0])
                    else:
                        mutex_vars.append(1 - model.input_vars[SAS_val][0])
                    
                if SAS_vars_goal[var_idx][val_idx] is not None:
                    mutex_group_goal.append(SAS_vars_goal[var_idx][val_idx])
                    SAS_val, not_negated = SAS_vars_goal[var_idx][val_idx]
                    if not_negated:
                        mutex_vars_goal.append(model.input_vars[SAS_val][0])
                    else:
                        mutex_vars_goal.append(1 - model.input_vars[SAS_val][0])
            
            # Save the keys of the mutex to the model
            model.mutexes.update([tuple(mutex_group), tuple(mutex_group_goal)])
                
            # Add mutex constraints
            model.model.addConstr(gp.quicksum(mutex_vars) <= 1)                
            model.model.addConstr(gp.quicksum(mutex_vars_goal) <= 1)
                
            i += 1
            if i >= num_mutex:
                break
            
            line_num += 3
            num_curr_mutex = int(lines[line_num])
    
    model.model.update()

def convert_name(atom, is_goal = False):
    '''
    Convert the format of the SAS variable into the form for the gurobi
    variables, tupled with whether (1) or not (0) the atom is negated.
    '''
    if atom.startswith("Atom"):
        not_negated = True
        strip_atom = atom[5:]    
    elif atom.startswith("NegatedAtom"):
        not_negated = False
        strip_atom = atom[12:]
    else:
        return None
        # raise KeyError(f"Do not know how to process {atom}")
    
    model_var = strip_atom[:strip_atom.index("(")]
    var_in_str = re.search("\(([^)]+)", strip_atom)
    if var_in_str is not None:
        var_in_list = var_in_str.group(1).split(', ')
        for obj in var_in_list:
            model_var += f"_{obj}"
    
    if is_goal:
        model_var = f"is_goal({model_var})"
    else:
        model_var = f"is_true({model_var})"

    return (model_var, not_negated)

def remove_pruned_props(model, SAS_vars, SAS_vars_goal):
    '''
    Some propositions are cancelled by FD but still appear in the ASNets input.
    Fix these values in the MILP model to whatever they are in the inital state
    (as this is what they will always be) and add them to a list so they can be
    excluded from any explanation.
    '''
    # Collate all of the SAS vars to determine the props were not pruned by FD
    not_pruned = {
        p[0]
        for var_vals in SAS_vars
        for p in var_vals
        if p is not None
    }
    not_pruned |= {
        p[0]
        for var_vals in SAS_vars_goal
        for p in var_vals
        if p is not None
    }
    all_props = set(model.input_vars.keys())
    model.pruned = all_props - not_pruned
    
    # Find the value from the initial state, as this will be fixed and be
    # constrained into the Gurobi model
    init_state = get_init_cstate(model.network.planner_exts)
    model.initialise_temp_constraints(init_state)
    
    for prop in model.pruned:
        key = prop[8:-1]
        
        if prop.startswith("is_true"):
            model.model.addConstr(
                model.input_vars[prop][0] == model._is_true[key]
            )
        elif prop.startswith("is_goal"):
            model.model.addConstr(
                model.input_vars[prop][0] == model._is_goal[key]
            )
            
        # Also delete from input vars because they are static
        # del model.input_vars[prop]
    
    model.remove_temp_constraints(reset=True)