import argparse
import logging
import time
import tensorflow as tf
from time import process_time, strftime

from asnets.interactive_network import NetworkInstantiator
from ASNets_compute_trajectory import roll_out_trajectory
from ASNets_model.gurobi_network import Model
from explanation_algorithms import subset_min_exp
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
    
def store_results(args, network, model, et_cpu, et_clock, states, actions, exp):
    '''Write to file results of the experiment'''
    # Create log file
    timestr = strftime("%Y%m%d-%H%M%S")
    file_name = args.path + "/" + network.single_problem.name + "-" + timestr + ".log"
    logging.basicConfig(filename=file_name, filemode="a", level=logging.INFO)
    
    # Get problem name
    logging.info(f"Problem path: {args.problem}")
    
    # Was the goal reached?
    logging.info(f"Goal reached/Finished in time: {int(states[-1].is_terminal)}")
    
    # Runtime (s)
    logging.info(f"CPU Runtime: {et_cpu}")
    logging.info(f"Runtime: {et_clock}")
    
    # Plan length (number of actions in the trajectory)
    logging.info(f"Plan length: {len(actions)}")
    
    # Size of explanation
    if exp:
        logging.info(f"Size of explanation: {len(exp)}")
    
    # Size of input
    logging.info(f"Size of input: {len(model.init_exp)}")
    
    # Number of calls to gurobi network
    logging.info(f"Number of calls: {model.model_calls}")
    
    # Runtime of each of the calls to the gurobi network
    logging.info(f"Runtime for each network call: {model.runtime}")

    logging.info(f"Tiebreaks used: {model.total_num_tiebreaking}")
    logging.info(f"Tiebreak runtime: {model.tiebreak_runtime}")
    
    # Size of the neural network
    logging.info(f"Number of inner modules ASNets: {model.size_network}")
    
    # Size of the Gurobi network
    num_vars = model.model.NumVars
    num_constrs = model.model.NumConstrs + model.model.NumQConstrs + \
        model.model.NumGenConstrs
    logging.info(
        f"Number of Gurobi variables: {num_vars} Constraints: {num_constrs}"
    )
    
    # Final explanation
    logging.info(f"Explanation: {model.exp}")

def main(args):
    st_cpu = process_time()
    st_clock = time.perf_counter()
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
        model = Model(network, domain, args.problem, args.use_mutex)
        
        if not states[-1].is_terminal or len(actions) == 0:
            et_cpu = process_time() - st_cpu
            et_clock = time.perf_counter() - st_clock
            store_results(args, network, model, et_cpu, et_clock, states, actions, None)
        elif args.type == "sub_min_exp":
            # Save the initial explanation to iterate over
            model.init_exp = [
                prop
                for prop in model.input_vars
                if not prop.startswith("is_goal")
            ]
                
            # Compute sensitivity formula for each proposition
            sensitivity = compute_sensitivity(model.network, model.init_exp, states, actions)
            removal_order = sorted(
                sensitivity,
                key=sensitivity.get
            )
            removal_order_reverse = [p for p in reversed(removal_order)]
                
            for order in [removal_order, removal_order_reverse]:
                st_cpu = process_time()
                st_clock = time.perf_counter()

                model = Model(network, domain, args.problem, args.use_mutex)
                
                exp = subset_min_exp(model, states, actions, order)
                et_cpu = process_time() - st_cpu
                et_clock = time.perf_counter() - st_clock
        
                store_results(args, network, model, et_cpu, et_clock, states, actions, exp)

parser = argparse.ArgumentParser()
parser.add_argument("weights")
parser.add_argument("domain")
parser.add_argument("problem")
parser.add_argument("--type", required=True, choices=(
    "sub_min_exp"
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
    main(parser.parse_args())