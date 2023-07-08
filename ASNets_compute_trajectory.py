'''
From asnets.scripts.sparse_activation_vis, runs the given policy on a problem.

Imported here to avoid having to install graph_tool.
'''

import numpy as np
import tensorflow as tf

from asnets.state_reprs import sample_next_state, get_init_cstate  # noqa

def roll_out_trajectory(policy, planner_exts, max_actions=150):
    """Runs the given policy from the initial state of a problem to the goal
    (or until the horizon given by `max_actions`)."""
    # see supervised.collect_paths for one example of how to do this
    cstates = [get_init_cstate(planner_exts)]
    enabled_acts_all = [cstates[-1].acts_enabled]
    act_layer_inputs_first = []
    act_layer_outputs = []
    prop_layer_outputs = []
    actions = []
    costs = []
    sess = tf.get_default_session()
    prob_meta = planner_exts.problem_meta
    while not cstates[-1].is_terminal and len(actions) < max_actions:
        this_state = cstates[-1]
        vec_state = [this_state.to_network_input()]
        next_act_dist_tensor, next_act_inputs, next_act_outputs, \
            next_prop_outputs = sess.run([
                policy.act_dist, policy.action_layer_input, policy.act_layers,
                policy.prop_layers
            ], feed_dict={policy.input_ph: vec_state})
        next_act_dist, = next_act_dist_tensor
        next_act_id = int(np.argmax(next_act_dist))
        act_layer_inputs_first.append(next_act_inputs)
        act_layer_outputs.append(next_act_outputs)
        prop_layer_outputs.append(next_prop_outputs)
        # we get BoundAction & add it to the trajectory
        # (use .unique_ident to convert it to something comprehensible)
        actions.append(prob_meta.bound_acts_ordered[next_act_id])
        next_state, cost = sample_next_state(cstates[-1], next_act_id,
                                             planner_exts)
        costs.append(cost)
        cstates.append(next_state)
        enabled_acts_all.append(next_state.acts_enabled)
    return cstates, actions, act_layer_inputs_first, act_layer_outputs, \
        prop_layer_outputs, costs, enabled_acts_all