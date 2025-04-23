import gymnasium as gym
from gymnasium import envs
from gymnasium.envs.registration import register
import sys
import os
import matplotlib.pyplot as plt
import time


# Adding python src code directory
sys.path.append(os.path.abspath("../python"))


from Ephemeris import Ephemeris
from StateVectorUtilities import polar_to_cartesian
from Hamiltonian_Control import Hamiltonian_Controller_TBT


# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")


def test_random_env_rest(env):
    # parameters
    num_trajs = 10

    # reset the evironment using set seed value and set counters
    seed_env = 42
    count = 0
    count_pos_mass_rate = 0
    count_solved = 0

    # track elapsed time
    start_time = time.time()
    start_time_last_traj = start_time

    # summary string array - initialize with header
    sa_report = []
    header = "Count,Delta Time (s),Fuel Used (kg),Bool Targeter Converged,Bool Positive Mass Rate"
    header = header + ",Lam1,Lam2,Lam3,Lam4,Lam5"
    sa_report.append(header)

    while count < num_trajs:
        count = count + 1
        seed_env = seed_env + 1

        eph = Ephemeris()
        init_observation, init_info = env.reset(seed=seed_env)

        r, theta, v_r, v_theta = init_observation[0:4]
        x, y, vx, vy = polar_to_cartesian(r, theta, v_r, v_theta)

        # ephemeris
        eph = Ephemeris()

        # extract some parameters of interest
        sun_rad = env.unwrapped.planet_radii[0]

        # compute Hamiltonian Solution
        input_TOF = 1.1 * 365.25 * 24 * 60 * 60

        H_controller = Hamiltonian_Controller_TBT(
            env, init_observation, init_info, input_TOF
        )

        # compute solution
        flag_solved, h_sol, eps, sol, log = H_controller.hamiltonian_solution_finder()

        # initialize to zero
        net_delta_m = 0.0
        if flag_solved:
            count_solved = count_solved + 1

            # write output ephemeris
            eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
                H_controller.generate_output_ephemeris(eph)
            )

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(arr_time, arr_rho)
            ax.set_title("Traj #" + str(count) + " Switching Function")
            plt.show()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(arr_time, arr_u)
            ax.set_title(
                "Traj #" + str(count) + " Spacecraft Thrust Throttle over Time"
            )
            plt.show()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(arr_time, arr_alpha_x, label="alpha_x")
            ax.plot(arr_time, arr_alpha_y, label="alpha_y")
            ax.set_title(
                "Traj #" + str(count) + " Alpha Vector (Maneuver Direction) over Time"
            )
            ax.legend()
            plt.show()

            # Ephemeris plotting
            sun_rad = 6.957e8
            sma_Earth = 149598023 * 1000  # m
            sma_Mars = 2.32495e8 * 1000  # m
            eph_out.plot_xy(sun_rad, "Traj #" + str(count))
            eph_out.plot_xy_ref_orbit(sma_Earth, "Earth Orbit")
            eph_out.plot_xy_ref_orbit(sma_Mars, "Mars Orbit")
            plt.show()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(arr_time, eph_out.arr_m)
            ax.set_title("Traj #" + str(count) + " Spacecraft Mass over Time")
            plt.show()

            # determine mass expenditure
            net_delta_m = eph_out.arr_m[0] - eph_out.arr_m[-1]

            arr_dm = []
            arr_zero = []
            flag_first_m = True
            flag_positive_mass_rate = False
            m_prev = 0.0
            t_prev = 0.0
            for i, m in enumerate(eph_out.arr_m):
                dm = m - m_prev
                et = arr_time[i]
                dt = et - t_prev
                if dm > 0.001 and not flag_first_m:
                    flag_positive_mass_rate = True
                    print("t: ", et, "Mass rate: ", dm / dt)

                if not flag_first_m:
                    arr_dm.append(dm)
                else:
                    arr_dm.append(0.0)

                arr_zero.append(0)
                m_prev = m
                t_prev = et
                flag_first_m = False

            # plot mass rate
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(arr_time, arr_dm)
            ax.plot(arr_time, arr_zero)
            ax.set_title("Traj #" + str(count) + " Mass Rate over Time")
            plt.show()

            if flag_positive_mass_rate:
                print("Warning - Positive Mass Rate: ", flag_positive_mass_rate)
                count_pos_mass_rate = count_pos_mass_rate + 1

        end_time = time.time()
        delta_time = end_time - start_time_last_traj
        start_time_last_traj = end_time
        string_traj_data = str(count) + "," + str(delta_time) + "," + str(net_delta_m)
        string_traj_data = (
            string_traj_data
            + ","
            + str(flag_solved)
            + ","
            + str(flag_positive_mass_rate)
        )
        string_traj_data = string_traj_data + "," + str(h_sol[0]) + "," + str(h_sol[1])
        string_traj_data = string_traj_data + "," + str(h_sol[2]) + "," + str(h_sol[3])
        string_traj_data = string_traj_data + "," + str(h_sol[4])

        sa_report.append(string_traj_data)

        print("Traj count: ", count)
        print("Fuel used: ", net_delta_m)
        print("Solution for initial co-states: ", h_sol)
        print("Final smoothing parameter used in solution generation: ", eps)
        count = count + 1
        seed_env = seed_env + 1
        print("Elapsed time: ", delta_time )
        print("flag_solved: ", flag_solved )
        print("")

    end_time = time.time()
    total_time = end_time - start_time
    time_per_traj = total_time / count
    
    print("------------------------------------------------------------------")
    print("")
    print("Summary")
    print("")
    print("Postive mass rate count: ", count_pos_mass_rate )
    print("Traj count: ", count )
    print("Targeter converged count: ", count_solved )
    print("Total elapsed time: ", total_time)
    print("Average time per traj: ", time_per_traj)
    print("")
    
    #report test summary to a file
    file_path = "..\\..\\data\\test_data\\test_random_TBT_transfer_report.csv"
    
    with open(file_path, "w") as f:
        for line in sa_report:
            f.write(line + "\n")
        
    f.close()
        
    
    print("------------------------------------------------------------------")
    print("")
    print("Summary")
    print("")
    print("Postive mass rate count: ", count_pos_mass_rate)
    print("Traj count: ", count)
    print("Targeter converged count: ", count_solved)
    print("Total elapsed time: ", total_time)
    print("Average time per traj: ", time_per_traj)
    print("")

    # report test summary to a file
    file_path = "..\\..\\data\\test_data\\test_random_TBT_transfer_report.csv"

    with open(file_path, "w") as f:
        for line in sa_report:
            f.write(line + "\n")

    f.close()


test_random_env_rest(env)
