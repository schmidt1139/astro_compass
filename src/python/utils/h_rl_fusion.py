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
    arr_r_tot = []
    r_tot = 0.0

    for i in range(ephem_H.num_vectors):
            
    
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

        action_in = [
            state_H[11],  # alpha x
            state_H[12],  # alpha y
            state_H[13]  # u
        ]  # zero thrust

        obs, info = env.set_state(state_in)
        obs, reward, done, truncated, info = env.step(action_in)

        reward_pos_comp = info['pos_r_component']
        reward_vel_comp = info['vel_r_component']
        reward_mass_comp = info['mass_r_component']

        r_tot += reward

        arr_elapsed_time.append(ephem_H.arr_et[i]/Constants.DAYS_TO_SEC)
        arr_rewards.append(reward)
        arr_pos_r_components.append(reward_pos_comp)
        arr_vel_r_components.append(reward_vel_comp)
        arr_mass_r_components.append(reward_mass_comp)
        arr_r_tot.append(r_tot)



    return [arr_elapsed_time, arr_rewards, arr_pos_r_components, arr_vel_r_components, arr_mass_r_components, arr_r_tot]