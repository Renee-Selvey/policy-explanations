import math
import os
import re
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from time import strftime
from pddl_parser.PDDL import PDDL_Parser

from asnets.state_reprs import get_init_cstate

from ASNets_model.ASNet_conversion import \
    add_enabled_var, create_layers, underscore_unique_ident
from ASNets_model.mutex_conversion import add_mutex_constrs, run_FD

# Epsilon for determining tie breaks for the chosen action - ensures it is
# chosen deterministically
EPSILON = 0.01

class Model:
    '''
    Model to represent an ASNet
    '''
    def __init__(
        self,
        network,
        domain_path,
        problem_path,
        use_mutex,
        verbose=False
    ):
        self.network = network
        self.weight_manager = self.network.instantiator.weight_manager
        self.domain_path = domain_path
        self.problem_path = problem_path
        self.prob_meta = self.network.policy._prob_meta
        
        # Adds mutex constraints if indicated
        self.use_mutex = use_mutex
        # A set of tuples to store each mutex
        self.mutexes = set()
        # A set of pruned propositions that can be removed from the explanation
        self.pruned = set()
        # A set of non-trivial actions (not cancelled from FD)
        self.non_trivial_actions = set()
        
        # Count the number of times the model has been called
        self.model_calls = 0
        # Record the length of time for each gurobi call
        self.runtime = []
        # Save the length of the neural network
        self.size_network = 0
        # Save the inital potential explanation
        self.init_exp = []
        # Save the final explanation
        self.exp = ""

        # Find the total number of layers in the network
        self.num_layers = len(self.weight_manager.prop_weights) \
            + len(self.weight_manager.act_weights)

        # Dictionaries to store the variables of the network that have modifiable
        # constraints
        self._out = {}
        self.enabled_vars = {}
        self._max_ind = None
        self._max_action = None
        self._maximal = None
        self._max_and_enabled = None
        self._eq_max = None

        # keys := input variable key, either:
        #       bound_prop
        #       is_goal(bound_prop)
        #       heuristic_data(bound_act/prop)[index]
        # values := [Gurobi var, bool] for whether or not these variables will
        #           be constrained in the explanation calculation
        self.input_vars = {}        
        # keys := input variable key, either:
        #       is_true(prop)
        #       is_goal(prop)
        #       heuristic_data(bound_act/prop)[index]
        # values := gurobi variable
        self._network_input = {}
        self._is_true = {}
        self._is_goal = {
            underscore_unique_ident(prop): prop in self.prob_meta.goal_props
            for prop in self.prob_meta.bound_props_ordered
        }
        self._heuristic_data = {}
        
        # Temporary constraints and parameters that are added then removed per 
        # call of the network
        self._temp_in_constraints = [] # Added and removed each call
        self._temp_out_constraints = [] # Added and removed each initialisation
        self._tiebreaking_constraints = [] # Added as required and removed each call
        
        self.needs_tiebreaking = False
        self.total_num_tiebreaking = 0
        self.tiebreak_runtime = []
        
        # If verbose, allow Gurobi's output
        if verbose:
            # Create new model
            self.model = gp.Model("network")
        else:
            # Suppresses gurobi's output
            env = gp.Env(empty=True)
            env.setParam("OutputFlag",1)
            env.setParam("LogToConsole", 0)
            env.start()
            self.model = gp.Model("network", env=env)
            timestr = strftime("%Y%m%d-%H%M%S")
            file_name = self.network.single_problem.name + "-" + timestr + ".log"
            self.model.setParam('LogFile', f'./logs/{file_name}')
        
        # Get from the PDDL domain and problem the ground actions of the problem
        parser = PDDL_Parser()
        parser.parse_domain(self.domain_path)
        parser.parse_problem(self.problem_path)

        self._ground_actions = {}
        for a in parser.actions:
            for ga in a.groundify(parser.objects, parser.types):
                key = f"{ga.name}"
                for pred in ga.parameters:
                    key += f"_{pred}"

                self._ground_actions[key] = ga
        
        self._create_model()
        
        # Save the length of the neural network
        self.size_network += len(self.input_vars) + len(self._out)

        # Model must be initialised
        self.initialised = False

    def _create_model(self):
        '''
        Creates all constraints that will remain unchanged per call
        '''
        # Get non-trivial actions from FD
        run_FD(self)
        
        # Get the ASNet module variables. This function also creates the 
        # constraints for module connections
        elu_in, elu_out = create_layers(self)
        self.size_network += len(elu_in)
        
        # We want to compute the ELU of all non-final layers
        elu_module_keys = list(elu_in.keys())
        # Add ELU inner layer variables, accessed by the ELU module keys
        t = self.model.addVars(elu_module_keys, vtype=GRB.CONTINUOUS, name="t")
        s = self.model.addVars(elu_module_keys, vtype=GRB.CONTINUOUS, name="s")
        s_neg = self.model.addVars(
            elu_module_keys, 
            vtype=GRB.CONTINUOUS, 
            lb=-GRB.INFINITY,
            ub=0, 
            name="s_neg"
        )
        e = self.model.addVars(
            elu_module_keys,
            vtype=GRB.CONTINUOUS,
            ub=1,
            name="e"
        )
        z = self.model.addVars(elu_module_keys, vtype=GRB.BINARY, name="z")

        # Add ELU constraints
        for neuron in elu_module_keys:           
            self.model.addConstr(elu_in[neuron] == t[neuron] - s[neuron])
            # One of s, t must be 0
            self.model.addGenConstrIndicator(z[neuron], True, t[neuron] <= 0)
            self.model.addGenConstrIndicator(z[neuron], False, s[neuron] <= 0)
            
            # e = exp(-s)
            self.model.addConstr(s_neg[neuron] == -s[neuron])
            self.model.addGenConstrExp(s_neg[neuron], e[neuron])
            # out = t + e^(-s) - 1
            self.model.addConstr(elu_out[neuron] == t[neuron] + e[neuron] - 1)

        # Add output constraints
        self._initialise_output_actions(self.model)
        
        # Calculates and add constraints to remove mutexes
        if self.use_mutex:
            add_mutex_constrs(self)
        
        # os.remove("output.sas")
        
        self.model.update()

    def is_explanation(self):
        '''
        Determines whether or not the model is feasible
        Model will be satisfied if the free variables of the input can be
        assigned such that it returns a different action with highest probability.
        
        If the model is satisfiable, then it is not an explanation.
        '''
        assert self.initialised, '''Need to initialise the temporary constraints 
            before calculating explanation'''

        exp = None

        # Add the temporary constraints and input variables
        self._add_temp_constraints()

        # self.model.setParam(GRB.Param.Presolve, 0)
        self.model.setParam(GRB.Param.FuncPieces, -1)
        self.model.setParam(GRB.Param.FuncPieceLength, 1e-5)
        self.model.setParam(GRB.Param.FuncPieceError, 1e-6)
        self.model.setParam(GRB.Param.IntFeasTol, 1e-9)
        self.model.write('mymodel.lp')
        self.model.optimize()
        self.runtime.append(self.model.Runtime)
        
        # If satisfiable, cannot be an explanation
        if self.model.Status == 3: # infeasible
            exp = True
            self.tiebreak_runtime.append(False)
        if self.model.Status == 2: # optimal solution found
            exp = False
            
            # for v in self.model.getVars():
            #     print(f"{v.VarName}, {v.X}")
            
            # Solution found - check for the need of tie-breaking
            # Get the output values              
            # enabled_output_ranked = sorted(
            #     self._out[k].X
            #     for k in self._out.keys() 
            #     if self.enabled_vars[k].X == 1
            # )
            
            # check = True
            # if len(enabled_output_ranked) <= 1:
            #     check = False
            
            # if check and math.isclose(enabled_output_ranked[-1] - enabled_output_ranked[-2], 0):
            #     print("Using tiebreaking")
            #     self.needs_tiebreaking = True
            #     self.total_num_tiebreaking += 1
            #     # Need to rerun the explanation with tiebreaking
            #     self.model.reset()
            #     self.add_tiebreaking_constraints()
            #     self.model.optimize()
                
            #     if self.model.Status == 3: # infeasible
            #         exp = True
            #     if self.model.Status == 2: # optimal solution found
            #         # for v in self.model.getVars():
            #         #     print(f"{v.VarName}, {v.X}")
            #         exp = False
                    
            #     self.tiebreak_runtime.append(self.model.Runtime)
            # else:
            #     self.tiebreak_runtime.append(False)
            
        # Remove the temporary constraints and input variables
        self.remove_temp_constraints()

        # Increment the number of calls to the model
        self.model_calls += 1
        self.model.reset()
        return exp

    def _initialise_output_actions(self, m):
        '''
        Add constraints involving the output actions. We are trying to constrain 
        so the model tries to choose a different action - i.e. the maximum of 
        the output action probabilities cannot be the network's chosen action.

        model: Gurobi network
        '''
        # Keep track of which index the 'is-enabled' heuristic data is stored in
        enabled_idx = \
            get_init_cstate(self.network.planner_exts)._aux_dat_interp.index(
                "is-enabled"
            )

        final_module_keys = list(self._out.keys())

        # Indicator variables for whether or not that action is enabled
        for act_mod in final_module_keys:
            key = re.sub("\[.*\]$", "", act_mod)
            enabled_var = f"heuristic_data({key})[{enabled_idx}]"

            # Only need to create a new variable if it is not already defined by
            # the input vars
            if enabled_var not in self._network_input.keys():
                add_enabled_var(self, enabled_var)

            self.enabled_vars[act_mod] = self._network_input[enabled_var]

        # Indicator variables for whether or not that action is the model's 
        # chosen action
        self._max_ind = m.addVars(
            final_module_keys,
            vtype=GRB.BINARY,
            name="max_ind"
        )

        # Add variable that is assigned the value of the chosen action
        self._max_action = m.addVar(
            vtype=GRB.CONTINUOUS, 
            lb=-GRB.INFINITY, 
            name="max_action"
        )

        # Commented out to remove q constraints
        # # Get max and min of the enabled actions to get the argmax
        # ub = m.addVar(
        #     vtype=GRB.CONTINUOUS, 
        #     lb=-GRB.INFINITY, 
        #     name="out_ub"
        # )
        # m.addConstr(ub == gp.max_(self._out.values()))

        # lb = m.addVar(
        #     vtype=GRB.CONTINUOUS, 
        #     lb=-GRB.INFINITY, 
        #     name="out_lb"
        # )
        # m.addConstr(lb == gp.min_(self._out.values()))

        # # Add chosen action constraints
        # for neuron in self._max_ind.keys():
        #     # Maximum of the output probabilities is the chosen action only if
        #     # it's enabled
        #     m.addConstr(
        #         ((ub - lb) * (1 - enabled_vars[neuron])) + self._max_action >= \
        #         self._out[neuron]
        #     )
        #     m.addConstr(
        #         self._max_action <= \
        #         self._out[neuron] + ((ub - lb)*(1 - self._max_ind[neuron]))
        #     )

        #     # The chosen action must be enabled
        #     m.addGenConstrIndicator(
        #         self._max_ind[neuron],
        #         True,
        #         enabled_vars[neuron] == 1
        #     )
        
        for neuron in self._max_ind.keys():
            m.addGenConstrIndicator(
                self._max_ind[neuron],
                True,
                self.enabled_vars[neuron] == 1,
            )
            
            m.addGenConstrIndicator(
                self.enabled_vars[neuron],
                True,
                self._max_action >= self._out[neuron]
            )
            
            m.addGenConstrIndicator(
                self._max_ind[neuron],
                True,
                self._max_action <= self._out[neuron]
            )

        # There can only be one chosen action and that action can only be chosen
        # if enabled
        m.addConstr(gp.quicksum(self._max_ind) == 1)

        m.update()

    def initialise_temp_constraints(self, state, action=None):
        '''
        Before calling is_explanation, we need to set the output constraints and
        create dicts of input information

        state: a state of the network of type CanonicalState

        action: (Optional) a BoundAction which is the model's chosen action
        '''
        self._is_true = {underscore_unique_ident(k): v for k,v in state.props_true}
        # Reshape the aux data to match the action shapes
        aux_data = np.reshape(
            state.aux_data, 
            (-1, self.prob_meta.num_acts, self.weight_manager.extra_dim)
        )[0]

        for act, data in zip(self.prob_meta.bound_acts_ordered, aux_data):
            self._heuristic_data[underscore_unique_ident(act)] = list(data)

        # Add constraints involving the chosen action (only need to happen once
        # per network initialisation)
        if action:
            chosen_act = self._max_ind[
                f"{underscore_unique_ident(action)}[{self.num_layers // 2},0]"
            ]
            # Don't choose this action
            constr = self.model.addConstr(chosen_act == 0)
            self._temp_out_constraints.append(constr)

        self.initialised = True

    def _add_temp_constraints(self):
        '''Add constraints involving model's input and output actions'''
        # Constrain the variables of the input that we don't want to be free
        for in_id, (in_var, to_use) in self.input_vars.items():
            if to_use:
                prop_name = re.search("\((.*)\)", in_id).group(1)
                if in_id.startswith("is_true"):
                    constr = self.model.addConstr(in_var == self._is_true[prop_name])
                    self._temp_in_constraints.append(constr)
                elif in_id.startswith("is_goal"):
                    constr = self.model.addConstr(in_var == self._is_goal[prop_name])
                    self._temp_in_constraints.append(constr)
                else: # heuristic_data
                    aux_data_ind = int(re.search("\[([0-9]+)\]", in_id).group(1))
                    constr = self.model.addConstr(
                        in_var == self._heuristic_data[prop_name][aux_data_ind]
                    )
                    self._temp_in_constraints.append(constr)

        self.model.update()
        
    def add_tiebreaking_constraints(self):  
        # self.needs_tiebreaking = True     
        final_module_keys = list(self._out.keys())
        
        # Indicator variables for whether or not that action is maximal - used
        # for tie breaking
        self._maximal = self.model.addVars(
            final_module_keys,
            vtype=GRB.BINARY,
            name="maximal"
        )
        
        # Auxillary variables for whether the action's output value is the same
        # as _max_action and enabled
        self._max_and_enabled = self.model.addVars(
            final_module_keys,
            vtype=GRB.BINARY,
            name="eq_max_and_enabled"
        )
        
        # Auxillary variables for whether the action's output value is the same
        # as _max_action
        self._eq_max = self.model.addVars(
            final_module_keys,
            vtype=GRB.BINARY,
            name="eq_max"
        )
        
        # Need to get the ordering of the actions from the network to determine
        # tie breaks
        num_layers = len(self.network.instantiator.weight_manager.prop_weights) \
            + len(self.network.instantiator.weight_manager.act_weights)
        final_layer = num_layers // 2
        acts_ordered = []
        for act in self.prob_meta.bound_acts_ordered:
            act_key = f"{underscore_unique_ident(act)}[{final_layer},0]"
            if act_key in self._max_ind.keys():
                acts_ordered.append(act_key)
    
        # Add constraints to determine tie breaks
        for i, neuron in enumerate(acts_ordered):
            self._tiebreaking_constraints.append(self.model.addGenConstrIndicator(
                self._eq_max[neuron],
                True,
                self._max_action - self._out[neuron] <= EPSILON
            ))
            
            self._tiebreaking_constraints.append(self.model.addGenConstrIndicator(
                self._eq_max[neuron],
                False,
                self._max_action - self._out[neuron] >= EPSILON
            ))
                        
            self._tiebreaking_constraints.append(self.model.addGenConstrAnd(
                self._max_and_enabled[neuron],
                [self._eq_max[neuron], self.enabled_vars[neuron]]
            ))
            
            self._tiebreaking_constraints.append(self.model.addGenConstrIndicator(
                self._max_and_enabled[neuron],
                True,
                self._maximal[neuron] == 1
            ))
            
            self._tiebreaking_constraints.append(self.model.addGenConstrIndicator(
                self._maximal[neuron],
                True,
                self._max_and_enabled[neuron] == 1
            ))
        
            # Get all actions before this neuron in the ordered list
            prev_actions = [acts_ordered[j] for j in range(i)]
            
            self._tiebreaking_constraints.append(self.model.addGenConstrIndicator(
                self._max_ind[neuron],
                True,
                self._maximal[neuron] - gp.quicksum(
                    self._maximal[a]
                    for a in prev_actions
                ) == 1
            ))

        self.model.update()

    def remove_temp_constraints(self, reset=False):
        '''
        Remove constraints and variables involving model's input.

        If `reset`, both the temporary constraints for the input and the
        temporary constraints for the output are removed. Otherwise, only the
        input temporary constraints will be removed.
        '''
        # Remove temporary input constraints
        for constr in self._temp_in_constraints:
            self.model.remove(constr)
        # Reinitialise temporary input constraints
        self._temp_in_constraints = []
        
        if self.needs_tiebreaking:
            # Remove tiebreaking constraints
            for constr in self._tiebreaking_constraints:
                self.model.remove(constr)
            # Reinitialise tiebreaking constraints
            self._tiebreaking_constraints = []
            
            # Remove tiebreaking variables
            for var in self._maximal.values():
                self.model.remove(var)
            self._maximal = None
            
            for var in self._max_and_enabled.values():
                self.model.remove(var)
            self._max_and_enabled = None
            
            for var in self._eq_max.values():
                self.model.remove(var)
            self._eq_max = None
            
        self.needs_tiebreaking = False

        if reset:
            # Set the use of all input variables to false
            self.input_vars.update(
                (k, [v[0], False]) 
                for k,v in self.input_vars.items()
            )

            # Reinitialise the input information
            self._is_true = {}
            self._heuristic_data = {}

            # Remove temporary output constraints
            for constr in self._temp_out_constraints:
                self.model.remove(constr)
            # Reinitialise temporary output constraints
            self._temp_out_constraints = []

            self.initialised = False

        self.model.update()

    def sanity_check(self, state, action):
        '''
        Performs two sanity checks:

        1. With all free variables, the model is feasible
        2. With no free variables, the model will choose the actual action
        '''
        # With all free variables, the model is feasible
        self.initialise_temp_constraints(state, action)
        assert not self.is_explanation(), "{} is an explanation"
        self.remove_temp_constraints(reset=True)
        
        print("checking numerical stability")
        
        # With no free variables, the model will choose the actual action
        self.initialise_temp_constraints(state, action)
        for var in self.input_vars:
            self.input_vars[var][1] = True

        self._add_temp_constraints()
        self.add_tiebreaking_constraints()
        # self.model.setParam(GRB.Param.Presolve, 0)
        self.model.setParam(GRB.Param.FuncPieces, -1)
        self.model.setParam(GRB.Param.FuncPieceLength, 1e-5)
        self.model.setParam(GRB.Param.FuncPieceError, 1e-6)
        self.model.setParam(GRB.Param.IntFeasTol, 1e-9)
        self.model.write("mymodel.lp")
        self.model.optimize()

        # if self.model.Status == 2:
        #     for v in self.model.getVars():
        #         print(f"{v.VarName}, {v.X}")
                
        # assert self.model.Status == 2, "Can't find any possible action"
        
        if self.model.Status == 2:
            chosen_act = \
                f"{underscore_unique_ident(action)}[{self.num_layers // 2},0]"
            
            if self._max_ind[chosen_act].X != 1:
                # Need to rerun with higher epsilon
                print("Wrong action chosen, increasing Epsilon")
                EPSILON = 0.01
                
                self.remove_temp_constraints(reset=True)
                
                self.initialise_temp_constraints(state, action)
                self._add_temp_constraints()
                self.add_tiebreaking_constraints()
                
                self.model.reset()
                self.model.optimize()
                
                assert self.model.Status == 3, "Skipping this problem"

        self.remove_temp_constraints()
        self.model.reset()