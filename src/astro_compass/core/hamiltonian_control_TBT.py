import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from astro_compass.constants.constants import Constants
from astro_compass.core.exceptions import SpacecraftCollisionException
from astro_compass.core.propagation import (
    Hamiltonian_EOM_TBT_v2,
    smoothing_function_homotopic,
    smoothing_function_tanh,
)
from astro_compass.envs.TwoBody_Orb2Orb_Transfer_Env_target import (
    TwoBody_Orb2Orb_Transfer_Env_target,
)
from astro_compass.utils.state_vector_utils import (
    cartesian_to_polar,
    compute_kep_velocities,
    non_dimensionalize,
)


class Hamiltonian_Controller_TBT:
    def extract_env_boundary_conditions(self):
        # extract initial conditions
        x0 = self.init_observation[0] * 1000  # x in m
        y0 = self.init_observation[1] * 1000  # y in m
        vx0 = self.init_observation[2] * 1000  # vx in m/s
        vy0 = self.init_observation[3] * 1000  # vy in m/s
        m_0 = self.init_observation[4]  # mass in kg

        mu = Constants.MU_SUN_M  # solar gravitational parameter in m^3/s^2

        # final target orbital elements
        self.a_target = self.init_observation[5] * 1000  # target semi-major axis in m
        self.e_target = self.init_observation[6]  # target eccentricity
        self.w_target = self.init_observation[7]  # target argument of periapsis in rad

        r_f = self.init_observation[5] * 1000  # final r in m
        r_dot_f = 0.0  # final radial r in m/s
        v_theta_f = (mu / r_f) ** 0.5  # final tangential vel in m/s

        # constants
        self.l_star = Constants.SMA_EARTH  # Earth SMA in m
        self.mu = mu  # gravitational parameter in m^3/s^2
        self.t_star = (self.l_star**3 / self.mu) ** 0.5  # non-dimensional time in s
        self.m_star = m_0  # kg
        g0 = 9.80665  # m/s^2

        # Parameters
        T_max = self.init_info["max_thrust"] * 1000  # max thrust in N
        ISP = self.init_info["ISP"]  # specific impulse of thruster in seconds

        # Create initial state array
        arr_y0 = np.array([x0, y0, vx0, vy0, m_0])

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

        # Unpack state vector
        arr_y0_nd, g0_nd, mu_nd, T_max_nd, ISP_nd, input_TOF_nd = nd_outputs
        self.arr_y0_nd = arr_y0_nd
        self.g0_nd = g0_nd
        self.mu_nd = mu_nd
        self.T_max_nd = T_max_nd
        self.ISP_nd = ISP_nd
        self.input_TOF_nd = input_TOF_nd

        # initial co-state guess
        lam_x0 = 0.1
        lam_y0 = 0.1
        lam_vx0 = 0.1
        lam_vy0 = 0.1
        lam_m0 = 0.1

        # Pack initial co-state vector
        self.arr_lam_0 = np.array([lam_x0, lam_y0, lam_vx0, lam_vy0, lam_m0])

        # Non-dimensionalize final boundary states
        self.r_f_nd = r_f / self.l_star
        self.r_dot_f_nd = r_dot_f / self.l_star * self.t_star
        self.v_theta_f_nd = v_theta_f / self.l_star * self.t_star

        # solution found flag (default to false)
        self.flag_solved = False

        self._log_controller_info("Boundary Conditions")
        self._log_controller_info(f"arr_y nd: {arr_y0_nd}")
        self._log_controller_info(f"t_star: {self.t_star}")
        self._log_controller_info("")
        self._log_controller_info(f"r_f_nd: {self.r_f_nd}")
        self._log_controller_info("r_dot_f_nd: " + str(self.r_dot_f_nd))
        self._log_controller_info("v_theta_f_nd: " + str(self.v_theta_f_nd))
        self._log_controller_info("a_target nd: " + str(self.a_target / self.l_star))
        self._log_controller_info("e_target nd: " + str(self.e_target))
        self._log_controller_info("w_target deg: " + str(np.rad2deg(self.w_target)))
        self._log_controller_info(
            "Initial co-state vector guess " + str(self.arr_lam_0)
        )

    def __init__(
        self,
        env: TwoBody_Orb2Orb_Transfer_Env_target,
        init_observation,
        init_info,
        input_TOF,
        **kwargs,
    ):
        # Targeter log string array
        self.log = []

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
        self.gamma = 0.5
        self.eps_threshold = 0.0001
        self.eps_0 = 1.0
        self.eps = self.eps_0
        self.max_k = 640
        self.root_tol = 1.0e-3
        self.root_tol_max = 0.001
        self.flag_constrain_u = True
        self.root_method = "lm"  # Choose from "hybr", "lm", "broyden1"
        self.root_max_iters = 16000
        self.smoothing_method = 0  # Choose from 0 (tanh), 1 (homotopic)
        self.flag_stop_targeting = False
        self.ivp_solve_rtol = 10 ** (-9)
        self.ivp_solve_atol = 10 ** (-12)
        self.shooting_iters = 0
        self.flag_report_live = False
        self.init_costate_guesses = 100
        self.timeout_per_trajectory = None

        # Check keyword args and override values
        allowed_kwargs = {
            "flag_report_live",
            "eps_threshold",
            "init_costate_guesses",
            "root_max_iters",
            "timeout_per_trajectory",
            "gamma",
            "root_tol",
        }
        self._log_controller_info("Hamiltonian Targeter Initialized")

        for key, val in kwargs.items():
            if key in allowed_kwargs:
                setattr(self, key, val)
                self._log_controller_info("kwarg " + key + " set to " + str(val))
            else:
                raise ValueError(f"Unknown keyword argument: {key}")

        # extract the state vector boundary conditions from the problem
        self.extract_env_boundary_conditions()

    def shooting_iteration(self, lam_guess_shooting, eps):
        # construct full state vector at t=0
        arr_full_y0 = np.hstack((self.arr_y0_nd, lam_guess_shooting))

        # define time span
        t_span = (0, self.input_TOF_nd)
        t_eval = np.linspace(*t_span, 1000)

        # prescribed boundary conditions for lambda_m and lambda_theta
        lam_m_f = 0.0

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
                eps,
                self.flag_constrain_u,
                self.smoothing_method,
            ]
        )

        # integrate forward in time
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

        if sol.status == -1:
            print(sol.message)
            raise Exception("Integration failed")

        # extract final cartesian state
        x_f_nd_p, y_f_nd_p, vx_f_nd_p, vy_f_nd_p, m_f_nd_p = sol.y[:5, -1]

        x_f_dim = x_f_nd_p * self.l_star
        y_f_dim = y_f_nd_p * self.l_star
        vx_f_dim = vx_f_nd_p * self.l_star
        vy_f_dim = vy_f_nd_p * self.l_star

        # extract final co-state
        lam_x_f_nd_p, lam_y_f_nd_p, lam_vx_f_nd_p, lam_vy_f_nd_p, lam_m_f_nd_p = sol.y[
            5:10, -1
        ]

        # convert state to polar coordinates
        r_f_nd_p, theta_f_nd_p, vr_f_nd_p, vtheta_f_nd_p = cartesian_to_polar(
            x_f_nd_p, y_f_nd_p, vx_f_nd_p, vy_f_nd_p
        )

        # residuals = np.array(
        #     [
        #         r_f_nd_p - self.r_f_nd,  # Final radius constraint
        #         vr_f_nd_p - self.r_dot_f_nd,  # Final radial velocity constraint
        #         vtheta_f_nd_p
        #         - self.v_theta_f_nd,  # Final tangential velocity constraint
        #         0.0,  # Co-state for theta shouldn't change
        #         lam_m_f_nd_p - lam_m_f,  # Final mass co-state should be 0
        #     ]
        # )

        r_target_nd, r_dot_target_nd, v_theta_target_nd = self.compute_residuals_comps(
            x_f_dim, y_f_dim
        )

        residuals = np.array(
            [
                r_f_nd_p - r_target_nd,  # Final radius constraint
                vr_f_nd_p - r_dot_target_nd,  # Final radial velocity constraint
                vtheta_f_nd_p
                - v_theta_target_nd,  # Final tangential velocity constraint
                0.0,  # Co-state for theta shouldn't change
                lam_m_f_nd_p - lam_m_f,  # Final mass co-state should be 0
            ]
        )

        self.shooting_iters = self.shooting_iters + 1

        # self._log_controller_info(f"Shooting iteration {self.shooting_iters}: Residuals = {residuals}")
        # self._log_controller_info(f"r_mag: {np.linalg.norm(residuals)}")

        return residuals

    def compute_residuals_comps(self, x_f_dim, y_f_dim):
        aol_current = np.arctan2(y_f_dim, x_f_dim)

        theta_current = aol_current - self.w_target
        theta_current = theta_current % (2 * np.pi)

        r_target = (
            self.a_target
            * (1 - self.e_target**2)
            / (1 + self.e_target * np.cos(theta_current))
        )
        r_dot_target, v_theta_current, v_tot_current = compute_kep_velocities(
            self.a_target, self.e_target, theta_current, self.mu
        )

        r_target_nd = r_target / self.l_star
        r_dot_target_nd = r_dot_target / self.l_star * self.t_star
        v_theta_target_nd = v_theta_current / self.l_star * self.t_star

        return r_target_nd, r_dot_target_nd, v_theta_target_nd

    def hamiltonian_root_finder(self, eps, lam_guess):
        try_max = 10
        try_count = 1
        flag_continue = True

        while flag_continue:
            # Call root finder method with the current eps (smoothing parameter)
            # and the current root finder tolerance value. If the root finder
            # function fails to reach a solution, the process is repeated with
            # a relaxed tolerance value. This process is repeated until the
            # a maximum try count is reached or if the

            if self.root_method == "lm":
                lam_sol = root(
                    self.shooting_iteration,
                    lam_guess,
                    args=(eps,),
                    method="lm",
                    options={"ftol": self.root_tol, "maxiter": self.root_max_iters},
                )
            elif self.root_method != "hybr":
                lam_sol = root(
                    self.shooting_iteration,
                    lam_guess,
                    eps,
                    tol=self.root_tol,
                    method=self.root_method,
                    options={"maxiter": self.root_max_iters},
                    jac=None,
                )

            else:
                lam_sol = root(
                    self.shooting_iteration,
                    lam_guess,
                    args=(eps,),
                    tol=self.root_tol,
                    method=self.root_method,
                    options={
                        "xtol": 1e-6,  # loosen tolerance
                        "maxfev": 5000,  # increase function evaluations
                    },
                )

            # self._log_controller_info("i: " + str(self.shooting_iters ) )
            # self._log_controller_info("Psi mag: " + str(np.linalg.norm(lam_sol.fun) ) + "\n" )

            # check if targeter is actually within tolerance if there is an early
            # exit
            if np.linalg.norm(lam_sol.fun) < self.root_tol:
                lam_sol.success = True

            if self.root_method != "broyden1":
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
        else:
            lam_solution = lam_sol.x
            self._log_controller_info("Lambda solution: " + str(lam_sol))

            if self.root_method != "broyden1":
                self._log_controller_info(str(fjac))
                self._log_controller_info("Jacobian condition number: " + str(cn))
                self._log_controller_info(
                    "Root tolerance reached: " + str(self.root_tol)
                )

            self._log_controller_info("Try count: " + str(try_count))
            self.flag_stop_targeting = True

        return lam_solution, lam_sol.fun

    def hamiltonian_solution_finder(self):
        # Initial smoothing parameters for the solution finder. The number of
        # smoothing iterations that has been performed is tracked with the k
        # counter. The initial epsilon value is taken from the object property
        # self.eps_0.
        k = 1
        eps = self.eps_0
        self.res_norm = np.inf

        # The first step is to check the initial co-state guess, if it does not
        # lie sufficiently close to the real solution and the root finder fails,
        # we re-try until a solution is achieved.
        arr_lam_sol_0 = self.check_initial_costate_guess()

        # provide initial co-state guess
        arr_lam_sol_0 = self.arr_lam_0
        arr_lam_sol_k = arr_lam_sol_0

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
            arr_lam_sol_k, res = self.hamiltonian_root_finder(eps, arr_lam_sol_0)

            # next initial guess is the previous solution
            arr_lam_sol_0 = arr_lam_sol_k
            res_norm = np.linalg.norm(res)
            self.res_norm = res_norm

            self._log_controller_info("arr_lam_sol_k: " + str(arr_lam_sol_k))
            self._log_controller_info("Residuals: " + str(res))
            self._log_controller_info("Residual norm: " + str(res_norm))
            self._log_controller_info("")

            # update k counter
            k = k + 1

            if self.flag_stop_targeting:
                break

        # assign co-state solution to Hamiltonian object after smoothing
        # iteration is complete.
        self.eps = eps
        self.arr_lam_sol = arr_lam_sol_k

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

        # integrate forward in time
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

        # assign solution to controller object and set solution flag to true
        self.final_sol = sol

        if sol.status == -1 or self.flag_stop_targeting:
            self._log_controller_info(sol.message)
            self.flag_solved = False
            self._log_controller_info("Targeter failed to converge")
            self._log_controller_info("Epsilon reached: " + str(self.eps))
            self._log_controller_info("Shooting iters: " + str(self.shooting_iters))
            self._log_controller_info("Root tol: " + str(self.root_tol))
        else:
            self.flag_solved = True

            if self.res_norm > self.root_tol:
                self._log_controller_info(
                    "Targeter didn't fully converge to desired tolerance"
                )
                self.flag_solved = False
                return self.flag_solved, self.arr_lam_sol, self.eps, sol, self.log
            else:
                self._log_controller_info("Targeter converged")
                self._log_controller_info("Shooting iters: " + str(self.shooting_iters))
                self._log_controller_info("Root tol: " + str(self.root_tol))
                self._log_controller_info("Final epsilon: " + str(self.eps))
                self._log_controller_info("Final residual norm: " + str(self.res_norm))
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

            # find what the target state should be
            state_in = [
                x_i,
                y_i,
                vx_i,
                vy_i,
                m_i,
                self.a_target,
                self.e_target,
                self.w_target,
            ]
            obs, info = self.env.set_state(state_in)
            state_out = self.env.get_cartesian_state()

            x_i_target = state_out[5]
            y_i_target = state_out[6]
            vx_i_target = state_out[7]
            vy_i_target = state_out[8]

            # time to go
            ttg_i = self.input_TOF - t_i

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
                x_i_target,
                y_i_target,
                vx_i_target,
                vy_i_target,
                ttg_i,
                alpha_vec[0],
                alpha_vec[1],
                u,
            )

            arr_u.append(u)
            arr_rho.append(rho)
            alpha_vec_x.append(alpha_vec[0])
            alpha_vec_y.append(alpha_vec[1])

        return ephemeris, arr_time, arr_u, arr_rho, alpha_vec_x, alpha_vec_y

    def check_initial_costate_guess(self):
        self._log_controller_info("Checking initial co-state guess")

        flag_good_first_guess = False
        counter_first_guess = 0
        max_iters = self.init_costate_guesses
        mean_co_state_guess = 0.0
        std_co_state_guess = 0.01
        len_co_state_guess = len(self.arr_lam_0)
        bias_co_states = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
        lam_guess = self.arr_lam_0

        # Set the seed for the random initial guess variation
        rng = np.random.default_rng(seed=42)

        while not flag_good_first_guess:
            counter_first_guess = counter_first_guess + 1

            if counter_first_guess > max_iters:
                raise Exception("Cannot find good initial co-state guess")

            # randomize the first guess if the first guess is no good
            if counter_first_guess > 1:
                lam_guess = rng.normal(
                    loc=mean_co_state_guess,
                    scale=std_co_state_guess,
                    size=len_co_state_guess,
                )

                # add bias array
                lam_guess = lam_guess + bias_co_states

            # try shooting iteration with the current guess
            try:
                lam_sol = root(
                    self.shooting_iteration,
                    lam_guess,
                    self.eps_0,
                    tol=self.root_tol,
                    jac=None,
                )
                success = lam_sol.success
            except SpacecraftCollisionException:
                success = False

            if success:
                self._log_controller_info(
                    "Attempt "
                    + str(counter_first_guess)
                    + "   Lambda: "
                    + str(lam_guess)
                    + " passed"
                )
                return lam_guess
            else:
                self._log_controller_info(
                    "Lambda: "
                    + str(counter_first_guess)
                    + "   lam_guess "
                    + str(lam_guess)
                    + " failed"
                )

    def _log_controller_info(self, info):
        self.log.append(info)

        if self.flag_report_live:
            print(info)
