import os
from utils.log_utils import log
from constants.constants import Constants
from core.hamiltonian_control_TBR import Hamiltonian_Controller_TBR, FirstGuessException
from core.ephemeris_v2 import Ephemeris_v2
from core.exceptions import SpacecraftCollisionException, LowMassException

def gen_Hamiltonian_trajectory(env, seed_traj, tof_scale, params, ephem_filename,
                                test_log=[],flag_report_live=False):

    #function to generate a single hamiltonian trajectory
    test_log = log(
        "Test Two-Body Rendezvous Hamiltonian Controller", test_log, flag_report_live
    )

    #initial conditions
    flag_solved = False
    obs, info = env.reset(seed=seed_traj)

    test_log = log("Initial observation vector\n", test_log, flag_report_live)
    test_log = log("x_nd: " + str(obs[0]), test_log, flag_report_live)
    test_log = log("y_nd: " + str(obs[1]), test_log, flag_report_live)
    test_log = log("vx_nd: " + str(obs[2]), test_log, flag_report_live)
    test_log = log("vy_nd: " + str(obs[3]), test_log, flag_report_live)
    test_log = log("mass_nd: " + str(obs[4]), test_log, flag_report_live)
    test_log = log("x_target_nd: " + str(obs[5]), test_log, flag_report_live)
    test_log = log("y_target_nd: " + str(obs[6]), test_log, flag_report_live)
    test_log = log("vx_target_nd: " + str(obs[7]), test_log, flag_report_live)
    test_log = log("vy_target_nd: " + str(obs[8]), test_log, flag_report_live)
    test_log = log("", test_log, flag_report_live)

    for item in info:
        test_log = log(f"{item}: {info[item]}", test_log, flag_report_live)

    test_log = log("", test_log, flag_report_live)

    #take the time of flight as the max period of the starting and target trajectories
    T_i = info["orbital_period_nd"]
    T_target = info["orbital_period_target_nd"]
    TOF = 0.0

    if (T_target > T_i):
        input_TOF = T_target * params["t_star"] * tof_scale
        test_log = log(f"Using TOF: {input_TOF/Constants.YEARS_TO_SEC} years", test_log, flag_report_live)    
    else:
        input_TOF = T_i * params["t_star"] * tof_scale
        test_log = log(f"Using TOF: {input_TOF/Constants.YEARS_TO_SEC} years", test_log, flag_report_live)    

    kwargs = {
        "flag_report_live": flag_report_live,
        "eps_threshold": params["eps_threshold"],
        "init_costate_guesses": params["init_costate_guesses"],
        "root_max_iters": params["root_max_iters"],
    }

    init_obs = obs
    init_obs[0] = obs[0] * params["l_star"] / 1000 #convert to km
    init_obs[1] = obs[1] * params["l_star"] / 1000 #convert to km
    init_obs[2] = obs[2] * params["l_star"] / params["t_star"] / 1000 #convert to km/s
    init_obs[3] = obs[3] * params["l_star"] / params["t_star"] / 1000 #convert to km/s
    init_obs[4] = obs[4] * params["m_star"]
    init_obs[5] = obs[5] * params["l_star"] / 1000 #convert to km
    init_obs[6] = obs[6] * params["l_star"] / 1000 #convert to km
    init_obs[7] = obs[7] * params["l_star"] / params["t_star"] / 1000 #convert to km/s
    init_obs[8] = obs[8] * params["l_star"] / params["t_star"] / 1000 #convert to km/s

    try:
        # compute Hamiltonian Solution
        H_controller = Hamiltonian_Controller_TBR(
            env, init_obs, info, input_TOF, **kwargs
        )

        # compute solution
        flag_solved, h_sol, eps, sol, log_hsl = H_controller.hamiltonian_solution_finder()

        test_log = log(
            "Hamiltonian solution found: " + str(flag_solved), test_log, flag_report_live
        )

        test_log = log("Hamiltonian solution details:\n", test_log, flag_report_live)

        for item in log_hsl:
            test_log = log(item, test_log, flag_report_live)

        # write output ephemeris
        eph = Ephemeris_v2()
        if flag_solved == True:

            eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
                H_controller.generate_output_ephemeris(eph)
            )
        else:
            raise FirstGuessException("Hamiltonian solution not found")
        
        test_log = log("Target initial x,y:", test_log, flag_report_live)
        test_log = log("x_i_target: " + str(init_obs[5]), test_log, flag_report_live)
        test_log = log("y_i_target: " + str(init_obs[6]), test_log, flag_report_live)
        
        file_name = os.path.join(params["data_path"], ephem_filename + ".txt")
        eph_out.write_to_file(file_name)
        test_log = log("Wrote test ephem to following path...", test_log, flag_report_live)
        test_log = log(file_name, test_log, flag_report_live)

        return flag_solved, test_log, eph_out
        
    except FirstGuessException as e:
        test_log = log(
            "Hamiltonian not found: " + str(e), test_log, flag_report_live
        )

        return flag_solved, test_log, None

    except SpacecraftCollisionException as e:
        test_log = log(
            "Spacecraft collision during trajectory generation: " + str(e), test_log, flag_report_live
        )

        return flag_solved, test_log, None
    
    except LowMassException as e:
        test_log = log(
            "Low spacecraft mass during trajectory generation: " + str(e), test_log, flag_report_live
        )

        return flag_solved, test_log, None