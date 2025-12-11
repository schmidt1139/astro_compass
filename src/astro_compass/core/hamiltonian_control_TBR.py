import time
import warnings

import numpy as np
from constants.constants import Constants
from core.propagation import (
    Hamiltonian_EOM_TBT_v2,
    smoothing_function_homotopic,
    smoothing_function_tanh,
)
from core.shooting_methods import Hamiltonian_Controller_TBR_Shooting
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from scipy.integrate import solve_ivp
from scipy.optimize import root

from astro_compass.utils.state_vector_utils import (
    non_dimensionalize,
)


class Hamiltonian_Controller_TBR(Hamiltonian_Controller_TBR_Shooting):
    def extract_env_boundary_conditions(self):
        # extract initial conditions
        x0 = self.init_observation[0] * 1000  # x in m
        y0 = self.init_observation[1] * 1000  # y in m
        vx0 = self.init_observation[2] * 1000  # vx in m/s
        vy0 = self.init_observation[3] * 1000  # vy in m/s
        m_0 = self.init_observation[4]  # mass in kg
        x_f = self.init_observation[5] * 1000  # final x in m
        y_f = self.init_observation[6] * 1000  # final y in m
        vx_f = self.init_observation[7] * 1000  # final vx in m/s
        vy_f = self.init_observation[8] * 1000  # final vy in m/s

        # constants
        self.l_star = Constants.SMA_EARTH  # Earth SMA in m
        self.t_star = (self.l_star**3 / self.mu) ** 0.5  # non-dimensional time in s
        self.m_star = m_0  # kg
        g0 = 9.80665  # m/s^2
        mu = self.mu  # gravitational parameter in m^3/s^2

        # Parameters
        T_max = self.init_info["max_thrust"]  # max thrust in N
        ISP = self.init_info["ISP"]  # specific impulse of thruster in seconds

        # Create initial and final state arrays
        arr_y0 = np.array([x0, y0, vx0, vy0, m_0])
        arr_yf = np.array([x_f, y_f, vx_f, vy_f, 0.0])

        # Non-Dimensionalize State Vector and Parameters
        nd_outputs = non_dimensionalize(
            arr_y0,
            g0,
            mu,
            T_max,
            ISP,
            self.input_TOF,
            self.l_star,
            self.m_star,
            self.t_star,
        )

        nd_outputs_f = non_dimensionalize(
            arr_yf,
            g0,
            mu,
            T_max,
            ISP,
            self.input_TOF,
            self.l_star,
            self.m_star,
            self.t_star,
        )

        # Unpack the initial and final state vectors
        arr_y0_nd, g0_nd, mu_nd, T_max_nd, ISP_nd, input_TOF_nd = nd_outputs
        arr_yf_nd, _, _, _, _, _ = nd_outputs_f

        self.arr_y0_nd = arr_y0_nd
        self.arr_yf_nd = arr_yf_nd
        self.g0_nd = g0_nd
        self.mu_nd = mu_nd
        self.T_max_nd = T_max_nd
        self.ISP_nd = ISP_nd
        self.input_TOF_nd = input_TOF_nd

        # initial co-state guess
        lam_x0 = 0.01
        lam_y0 = 0.01
        lam_vx0 = 0.01
        lam_vy0 = 0.01
        lam_m0 = 0.01

        # Pack initial co-state vector
        self.arr_lam_0 = np.array([lam_x0, lam_y0, lam_vx0, lam_vy0, lam_m0])

        # solution found flag (default to false)
        self.flag_solved = False

        self._log_controller_info("Boundary Conditions")
        self._log_controller_info(f"x_0_nd: {self.arr_y0_nd[0]}")
        self._log_controller_info(f"y_0_nd: {self.arr_y0_nd[1]}")
        self._log_controller_info(f"vx_0_nd: {self.arr_y0_nd[2]}")
        self._log_controller_info(f"vy_0_nd: {self.arr_y0_nd[3]}")
        self._log_controller_info(f"t_star: {self.t_star}")
        self._log_controller_info("")
        self._log_controller_info(f"x_f_nd: {self.arr_yf_nd[0]}")
        self._log_controller_info(f"y_f_nd: {self.arr_yf_nd[1]}")
        self._log_controller_info(f"vx_f_nd: {self.arr_yf_nd[2]}")
        self._log_controller_info(f"vy_f_nd: {self.arr_yf_nd[3]}")
        self._log_controller_info(
            "Initial co-state vector guess " + str(self.arr_lam_0)
        )

    def __init__(
        self,
        env: TwoBodyRendezvous_Env,
        init_observation,
        init_info,
        input_TOF,
        **kwargs,
    ):
        # Targeter log string array
        self.log = []

        self.start_time = time.time()
        self.log.append(
            f"Initialization started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.env = env  # The Two body transfer gym environment
        self.init_observation = init_observation  # The initial state of the env
        self.init_info = init_info  # Initial env info dict
        self.input_TOF = input_TOF  # User input time of flight [s]

        # Smoothing parameters
        # eps_threshold: The min value of smoothing parameter needed to reach a solution
        # gamma: The value to multiply eps by to gradually decrease it to eps_threshold
        # eps_0: The value of epsilon to start at
        # eps: The current value of epsilon
        # max_k: The maximum number of smoothing iterations to perform (ensures exit)
        # root_tol: The root finder function zero tolerance (increased if solver struggles)
        # root_tol_max: The max root tolerance, if this value is reached and there
        # is still no convergence the targeting procedure fails.
        self.mu = init_info["mu"]  # gravitational parameter in m^3/s^2
        self.gamma = 0.5
        self.eps_threshold = 0.0001
        self.eps_0 = 1.0
        self.eps = self.eps_0
        self.max_k = 640
        self.root_tol = 1.0e-3
        self.root_tol_max = 1.0e-3
        self.flag_constrain_u = True
        self.root_method = "lm"  # Choose from "hybr", "lm", "broyden1"
        self.root_max_iters = 400
        self.smoothing_method = 0  # Choose from 0 (tanh), 1 (homotopic)
        self.flag_stop_targeting = False
        self.rtol_explore = 1e-7
        self.atol_explore = 1e-10
        self.ivp_solve_rtol = self.rtol_explore
        self.ivp_solve_atol = self.atol_explore
        self.rtol_final = 1e-9
        self.atol_final = 1e-12
        self.shooting_iters = 0
        self.flag_report_live = False
        self.init_costate_guesses = 16
        self.timeout_per_trajectory = None  # in seconds, None for no timeout
        self.flag_initial_costate_found = False

        # Check keyword args and override values
        allowed_kwargs = {
            "flag_report_live",
            "eps_threshold",
            "init_costate_guesses",
            "root_max_iters",
            "gamma",
            "timeout_per_trajectory",
            "rtol_explore",
            "atol_explore",
            "ivp_solve_rtol",
            "ivp_solve_atol",
            "root_tol",
            "root_tol_max",
        }

        for key, val in kwargs.items():
            if key in allowed_kwargs:
                setattr(self, key, val)
                self._log_controller_info("kwarg " + key + " set to " + str(val))
            else:
                raise ValueError(f"Unknown keyword argument: {key}")

        self._log_controller_info("Hamiltonian Targeter Initialized")

        # report input kwargs
        for key, val in kwargs.items():
            if key in allowed_kwargs:
                self._log_controller_info("kwarg " + key + " set to " + str(val))
            else:
                raise ValueError(f"Unknown keyword argument: {key}")

        # extract the state vector boundary conditions from the problem
        self.extract_env_boundary_conditions()

    def hamiltonian_root_finder(self, eps, lam_guess):
        try_max = 10
        try_count = 1
        flag_continue = True

        while flag_continue:
            # Call Levenberg-Marquardt root finder
            lam_sol = root(
                self.shooting_iteration,
                lam_guess,
                args=(eps,),
                method="lm",
                options={"ftol": self.root_tol, "maxiter": self.root_max_iters},
            )

            # Calculate actual residual norm
            residual_norm = np.linalg.norm(lam_sol.fun)

            # Strictly enforce residual tolerance - override solver's success flag
            if residual_norm < self.root_tol:
                lam_sol.success = True
                self._log_controller_info(
                    f"Root finding converged with residual: {residual_norm:.4e}"
                )
            else:
                # Reject solution if residual is too large
                lam_sol.success = False

            # Get Jacobian condition number for diagnostics
            fjac = lam_sol.fjac
            cn = np.linalg.cond(fjac)

            if lam_sol.success:
                flag_continue = False

            elif (try_count < try_max) and self.root_tol < self.root_tol_max:
                self.root_tol = self.root_tol * 10
                try_count = try_count + 1
                self._log_controller_info(
                    f"Increasing root tolerance value: {self.root_tol:.4e}"
                )

            else:
                self._log_controller_info(
                    "Maximum attempts reached for root finding method"
                )
                self._log_controller_info("self.root_tol: " + str(self.root_tol))
                flag_continue = False

        # The throttle should be constained after the first iteration. The cap
        # on the throttle is lifted to get an initial solution
        if not self.flag_constrain_u:
            self.flag_constrain_u = True

        # Check if the solution was successful
        if lam_sol.success:
            lam_solution = lam_sol.x
            residual_mag = np.linalg.norm(lam_sol.fun)
            iters_taken = lam_sol.nfev
        else:
            lam_solution = lam_sol.x
            residual_mag = np.linalg.norm(lam_sol.fun)
            self._log_controller_info("Lambda solution: " + str(lam_sol))
            iters_taken = lam_sol.nfev

            if self.root_method != "broyden1":
                self._log_controller_info(str(fjac))
                self._log_controller_info("Jacobian condition number: " + str(cn))
                self._log_controller_info(
                    "Root tolerance reached: " + str(self.root_tol)
                )

            self._log_controller_info("Try count: " + str(try_count))
            self.flag_stop_targeting = True

        return lam_solution, residual_mag, iters_taken

    def hamiltonian_solution_finder(self):
        # Initial smoothing parameters for the solution finder. The number of
        # smoothing iterations that has been performed is tracked with the k
        # counter. The initial epsilon value is taken from the object property
        # self.eps_0.
        k = 1
        eps = self.eps_0

        # The first step is to check the initial co-state guess, if it does not
        # lie sufficiently close to the real solution and the root finder fails,
        # we re-try until a solution is achieved.
        arr_lam_sol_0 = self.check_initial_costate_guess()

        # Use the validated initial guess from check_initial_costate_guess()
        # DO NOT overwrite with self.arr_lam_0!
        arr_lam_sol_k = arr_lam_sol_0

        # determine initial boundary values for co-states
        arr_lam_sol_k, residual_mag, f_iters = self.hamiltonian_root_finder(
            eps, arr_lam_sol_0
        )

        self._log_controller_info("k_max: " + str(self.max_k))
        self._log_controller_info("eps0: " + str(eps))
        self._log_controller_info("")

        while (k <= self.max_k) and (eps > self.eps_threshold):
            # update/decrease epsilon by gamma factor if it is not the first
            # iteration
            if k != 1:
                eps = eps * self.gamma

            self._log_controller_info("k: " + str(k))
            self._log_controller_info("gamma_k: " + str(self.gamma))
            self._log_controller_info("eps_k: " + str(eps))

            # determine initial boundary values for co-states
            arr_lam_sol_k, residual_mag, f_iters = self.hamiltonian_root_finder(
                eps, arr_lam_sol_0
            )

            # next initial guess is the previous solution
            arr_lam_sol_0 = arr_lam_sol_k

            time_elapsed = time.time() - self.start_time
            self._log_controller_info(f"Time elapsed: {time_elapsed:.2f} seconds")
            self._log_controller_info("arr_lam_sol_k: " + str(arr_lam_sol_k))
            self._log_controller_info("root f evals: " + " " + str(f_iters))
            self._log_controller_info("Residual mag: " + str(residual_mag))
            self._log_controller_info("Root tol: " + str(self.root_tol))
            self._log_controller_info(
                f"flag_constrain_u at iteration: {self.flag_constrain_u}"
            )
            self._log_controller_info("")

            # update k counter
            k = k + 1

            if self.flag_stop_targeting:
                break

        # assign co-state solution to Hamiltonian object after smoothing
        # iteration is complete.
        self.eps = eps
        self.arr_lam_sol = arr_lam_sol_k

        # Log the converged parameters for debugging
        self._log_controller_info("=== Final Integration Parameters ===")
        self._log_controller_info(f"eps: {self.eps:.6e}")
        self._log_controller_info(f"flag_constrain_u: {self.flag_constrain_u}")
        self._log_controller_info(f"arr_lam_sol: {self.arr_lam_sol}")
        self._log_controller_info(f"Residual from last iteration: {residual_mag:.4e}")

        # construct full state vector at t=0
        arr_full_y0 = np.hstack((self.arr_y0_nd, self.arr_lam_sol))

        # define time span
        t_span = (0, self.input_TOF_nd)
        t_eval = np.linspace(*t_span, 1000)

        # set up parameter array
        params = np.array(
            [
                self.mu_nd,
                self.T_max_nd,
                self.ISP_nd,
                self.l_star,
                self.m_star,
                self.t_star,
                self.g0_nd,
                self.eps,
                self.flag_constrain_u,
                self.smoothing_method,
            ]
        )

        # integrate forward in time - catch step size warning as exception
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="Required step size is less than spacing between numbers",
            )

            try:
                sol = solve_ivp(
                    Hamiltonian_EOM_TBT_v2,
                    t_span,
                    arr_full_y0,
                    method="RK45",
                    args=(params,),
                    t_eval=t_eval,
                    rtol=self.ivp_solve_rtol,
                    atol=self.ivp_solve_atol,
                )
            except UserWarning as w:
                raise Exception(
                    f"Integration failed in final ephemeris generation: {str(w)}"
                ) from w

        # assign solution to controller object and set solution flag to true
        self.final_sol = sol

        # Verify final boundary conditions match targets
        x_f_final = sol.y[0, -1]
        y_f_final = sol.y[1, -1]
        vx_f_final = sol.y[2, -1]
        vy_f_final = sol.y[3, -1]
        lam_m_f_final = sol.y[9, -1]

        final_residual = np.array(
            [
                x_f_final - self.arr_yf_nd[0],
                y_f_final - self.arr_yf_nd[1],
                vx_f_final - self.arr_yf_nd[2],
                vy_f_final - self.arr_yf_nd[3],
                lam_m_f_final - 0.0,
            ]
        )
        final_residual_norm = np.linalg.norm(final_residual)

        if sol.status == -1 or self.flag_stop_targeting:
            elapsed_time = time.time() - self.start_time
            self._log_controller_info(f"Total time elapsed: {elapsed_time:.2f} seconds")
            self._log_controller_info(sol.message)
            self.flag_solved = False
            self._log_controller_info("Targeter failed to converge")
            self._log_controller_info("Epsilon reached: " + str(self.eps))
            self._log_controller_info("Shooting iters: " + str(self.shooting_iters))
            self._log_controller_info("Root tol: " + str(self.root_tol))
            self._log_controller_info(
                f"Final integration residual: {final_residual_norm:.4e}"
            )
        else:
            elapsed_time = time.time() - self.start_time
            self._log_controller_info(f"Total time elapsed: {elapsed_time:.2f} seconds")
            self.flag_solved = True
            self._log_controller_info("Targeter converged")
            self._log_controller_info("Shooting iters: " + str(self.shooting_iters))
            self._log_controller_info("Root tol: " + str(self.root_tol))
            self._log_controller_info(
                f"Final integration residual: {final_residual_norm:.4e}"
            )
            self._log_controller_info(
                f"Final state: x={x_f_final:.6f}, y={y_f_final:.6f}, vx={vx_f_final:.6f}, vy={vy_f_final:.6f}"
            )
            self._log_controller_info(
                f"Target state: x={self.arr_yf_nd[0]:.6f}, y={self.arr_yf_nd[1]:.6f}, vx={self.arr_yf_nd[2]:.6f}, vy={self.arr_yf_nd[3]:.6f}"
            )

        return self.flag_solved, self.arr_lam_sol, self.eps, sol, self.log

    def generate_output_ephemeris(self, ephemeris):
        # only write ephemeris if the controller has found a solution
        if not self.flag_solved:
            raise Exception("Controller has not solved, cannot write ephemeris")

        # extract time and state variables from solution
        arr_time = self.final_sol.t
        arr_u = []
        arr_rho = []
        alpha_vec_x = []
        alpha_vec_y = []

        # step through states and add to ephem object
        for index, t in enumerate(arr_time):
            # states
            t_i = t * self.t_star
            x_i = self.final_sol.y[0, index] * self.l_star
            y_i = self.final_sol.y[1, index] * self.l_star
            vx_i = self.final_sol.y[2, index] * self.l_star / self.t_star
            vy_i = self.final_sol.y[3, index] * self.l_star / self.t_star
            m_i_nd = self.final_sol.y[4, index]
            m_i = m_i_nd * self.m_star
            x_target_i = self.arr_yf_nd[0] * self.l_star
            y_target_i = self.arr_yf_nd[1] * self.l_star
            vx_target_i = self.arr_yf_nd[2] * self.l_star / self.t_star
            vy_target_i = self.arr_yf_nd[3] * self.l_star / self.t_star
            TTG_i = (self.input_TOF_nd - t) * self.t_star

            # co-states
            lam_vx_i = self.final_sol.y[7, index]
            lam_vy_i = self.final_sol.y[8, index]
            lam_m_i = self.final_sol.y[9, index]

            lam_v_vec = np.array([lam_vx_i, lam_vy_i])
            lam_v_mag = np.linalg.norm(lam_v_vec)

            # Find alpha vector
            alpha_vec = -lam_v_vec / lam_v_mag

            # Switching function
            rho = lam_m_i + self.ISP_nd * self.g0_nd * lam_v_mag / m_i_nd - 1

            if self.eps == 0.0:
                if rho >= 0:
                    u = 1.0
                else:
                    u = 0.0
            else:
                # check the smoothing method
                if self.smoothing_method == 0:
                    u = smoothing_function_tanh(rho, self.eps)
                elif self.smoothing_method == 1:
                    u = smoothing_function_homotopic(
                        rho, self.eps, self.flag_constrain_u
                    )

            lam_v_vec = np.array([lam_vx_i, lam_vy_i])
            lam_v_mag = np.linalg.norm(lam_v_vec)

            ephemeris.add_data(
                t_i,
                x_i,
                y_i,
                vx_i,
                vy_i,
                m_i,
                x_target_i,
                y_target_i,
                vx_target_i,
                vy_target_i,
                TTG_i,
                alpha_vec[0],
                alpha_vec[1],
                u,
            )

            arr_u.append(u)
            arr_rho.append(rho)
            alpha_vec_x.append(alpha_vec[0])
            alpha_vec_y.append(alpha_vec[1])

        return ephemeris, arr_time, arr_u, arr_rho, alpha_vec_x, alpha_vec_y

    def _log_controller_info(self, info):
        self.log.append(info)

        if self.flag_report_live:
            print(info)
