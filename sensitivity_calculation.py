from copy import deepcopy
import tensorflow as tf

from ASNets_model.ASNet_conversion import underscore_unique_ident


def compute_sensitivity(network, init_exp, states, actions):
    '''Compute sensitivity formula for each proposition
    \pi^{a_i}(s_i[v]) - max_{a \in A(s_i) \neg a_i} \pi^a(s_i[v])'''
    
    sensitivity_dict = {
        prop: None
        for prop in init_exp
    }
    
    for l in sensitivity_dict.keys():
        act_sensitivity = []
        for state, action in zip(states, actions):
            # Deepcopy the actual versions so we don't mess up the experiments
            props_true_copy = tuple(x for x in state.props_true)
            goal_props_copy = tuple(x for x in network.policy._prob_meta.goal_props)
            
            if l.startswith("is_true"):
                # Strip is_true
                prop_key = l[8:-1]
                
                # Swap the prop value
                for i, (curr_prop, value) in enumerate(state.props_true):
                    if underscore_unique_ident(curr_prop) == prop_key:                       
                        props_true_list = list(list(x) for x in state.props_true)
                        props_true_list[i][1] = not value
                        state.props_true = tuple(tuple(x) for x in props_true_list)
                        break               

            else: # will be is_goal
                # Strip is_goal
                prop_key = l[8:-1]
                
                props_in_goal = [
                    underscore_unique_ident(prop)
                    for prop in network.policy._prob_meta.goal_props
                ]
                if prop_key in props_in_goal: # remove from goal props
                    network.policy._prob_meta.goal_props = tuple(
                        prop
                        for prop in network.policy._prob_meta.bound_props_ordered
                        if prop in network.policy._prob_meta.goal_props and \
                            underscore_unique_ident(prop) != prop_key
                    )
                else: # add into goal props
                    network.policy._prob_meta.goal_props = tuple(
                        prop
                        for prop in network.policy._prob_meta.bound_props_ordered
                        if prop in network.policy._prob_meta.goal_props or \
                            underscore_unique_ident(prop) == prop_key
                    )
            
            sess = tf.get_default_session()
            vec_state = [state.to_network_input()]
            next_act_dist_tensor, _, _, _ = sess.run([
                    network.policy.act_dist, network.policy.action_layer_input,
                    network.policy.act_layers, network.policy.prop_layers
                ], feed_dict={network.policy.input_ph: vec_state})
            next_act_dist, = next_act_dist_tensor
                
            # Get action value   
            for i, (curr_act, _) in enumerate(state.acts_enabled):
                if curr_act == action:
                    action_probability = next_act_dist[i]
                    max_remaining_acts = max([
                        next_act_dist[j]
                        for j in range(len(next_act_dist))
                        if i != j
                    ])
                    
                    act_sensitivity.append(action_probability - max_remaining_acts)
                    break
                        
            sensitivity_dict[l] = min(act_sensitivity)
            
            # Return back to the old values afterwards
            state.props_true = tuple(x for x in props_true_copy)
            network.policy._prob_meta.goal_props = tuple(x for x in goal_props_copy)
    
    return sensitivity_dict