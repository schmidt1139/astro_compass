from constants.constants import Constants
import numpy as np

def calc_rewards_from_H_ephem(ephem_H, env, params):

    test_log = {}
    model = None  # Placeholder, as model is not used in this function
    #arr_elapsed_time = np.array(ephem_H.arr_et)/Constants.DAYS_TO_SEC
    arr_elapsed_time = []
    arr_rewards = []
    arr_pos_r_components = []
    arr_vel_r_components = []
    arr_mass_r_components = []
    arr_throttle_r_components = []
    arr_time_r_components = []
    arr_r_tot = []
    arr_position_res = []
    arr_target_x_current = []
    arr_target_y_current = []
    arr_target_vx_current = []
    arr_target_vy_current = []
    arr_x_current = []
    arr_y_current = []
    arr_ttg = []
    arr_terminated = []
    arr_truncated = []

    r_tot = 0.0

    number_of_vectors = params.get("number_of_vectors_plot", ephem_H.num_vectors)

    if number_of_vectors < ephem_H.num_vectors:
        pass
    else:
        number_of_vectors = ephem_H.num_vectors
        

    for i in range(number_of_vectors):
            
    
        state_H = ephem_H.get_vector_at_index(i)

        obs, info = env.reset()


        # Set environment state to Hamiltonian ephemeris state
        state_in = [
            state_H[1],  # x
            state_H[2],  # y
            state_H[3],  # vx
            state_H[4],  # vy
            state_H[5],  # m
            state_H[6],  # x_target
            state_H[7],  # y_target
            state_H[8],  # vx_target
            state_H[9],  # vy_target
            state_H[10]   # ttg
        ]

        # convert action if necessary

        action_in = [
            state_H[13],  # u
            state_H[11],  # alpha x
            state_H[12]  # alpha y
        ]

        unwrapped_env = env.unwrapped
        obs, info = unwrapped_env.set_state(state_in)
        obs, reward, done, truncated, info = unwrapped_env.step(action_in)

        reward_pos_comp = info['pos_r_component']
        reward_vel_comp = info['vel_r_component']
        position_res = info['pos_residual']
        x_target_current = info['target_x_current']
        y_target_current = info['target_y_current']
        vx_target_current = info['target_vx_current']
        vy_target_current = info['target_vy_current']

        if 'mass_r_component' in info:
            reward_mass_comp = info['mass_r_component']
        else:
            reward_mass_comp = 0.0

        if 'throttle_r_component' in info:
            reward_throttle_comp = info['throttle_r_component']
        else:
            reward_throttle_comp = 0.0

        if 'time_r_component' in info:
            reward_time_comp = info['time_r_component']
        else:
            reward_time_comp = 0.0

        if 'terminated' in info:
            if info['terminated']:
                arr_terminated.append(1)
            else:
                arr_terminated.append(0)
        else:
            arr_terminated.append(0)

        r_tot += reward

        arr_elapsed_time.append(ephem_H.arr_et[i]/Constants.DAYS_TO_SEC)
        arr_rewards.append(reward)
        arr_pos_r_components.append(reward_pos_comp)
        arr_vel_r_components.append(reward_vel_comp)
        arr_mass_r_components.append(reward_mass_comp)
        arr_throttle_r_components.append(reward_throttle_comp)
        arr_time_r_components.append(reward_time_comp)
        arr_r_tot.append(r_tot)
        arr_position_res.append(position_res)
        arr_target_x_current.append(x_target_current)
        arr_target_y_current.append(y_target_current)
        arr_target_vx_current.append(vx_target_current)
        arr_target_vy_current.append(vy_target_current)
        arr_ttg.append(state_H[10]/86400)
        arr_x_current.append(state_H[1])
        arr_y_current.append(state_H[2])


    return [arr_elapsed_time, arr_rewards, arr_pos_r_components, 
            arr_vel_r_components, arr_mass_r_components, arr_throttle_r_components, 
            arr_time_r_components, arr_r_tot, arr_position_res, arr_target_x_current, 
            arr_target_y_current, arr_target_vx_current, arr_target_vy_current,
            arr_ttg, arr_x_current, arr_y_current, arr_terminated]