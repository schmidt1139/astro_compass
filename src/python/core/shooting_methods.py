import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from core.propagation import (
    Hamiltonian_EOM_TBT_v2
)
from core.exceptions import FirstGuessException
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.hamiltonian_control_TBR import Hamiltonian_Controller_TBR


class Hamiltonian_Controller_TBR_Shooting:

    if TYPE_CHECKING:
        # Declare attributes that exist in the main class
        flag_solved: bool
        shooting_iters: int
        eps: float
        root_tol: float
        log: list
        flag_report_live: bool
        arr_y0_nd: np.ndarray
        arr_yf_nd: np.ndarray
        input_TOF_nd: float
        mu_nd: float
        T_max_nd: float
        ISP_nd: float
        l_star: float
        m_star: float
        t_star: float
        g0_nd: float
        ivp_solve_rtol: float
        ivp_solve_atol: float
        flag_constrain_u: bool
        smoothing_method: str
        arr_lam_0: np.ndarray
        init_costate_guesses: int

        def _log_controller_info(self, info: str) -> None: ...

    def shooting_iteration(self, lam_guess_shooting, eps):
        # construct full state vector at t=0
        arr_full_y0 = np.hstack((self.arr_y0_nd, lam_guess_shooting))

        # unpack targets
        x_f_target_nd = self.arr_yf_nd[0]
        y_f_target_nd = self.arr_yf_nd[1]
        vx_f_target_nd = self.arr_yf_nd[2]
        vy_f_target_nd = self.arr_yf_nd[3]

        # define time span
        t_span = (0, self.input_TOF_nd)
        t_eval = np.linspace(*t_span, 1000)

        # prescribed boundary conditions for lambda_m
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
        x_f_nd_p, y_f_nd_p, vx_f_nd_p, vy_f_nd_p, _ = sol.y[:5, -1]

        # extract final co-state
        _, _, _, _, lam_m_f_nd_p = sol.y[
            5:10, -1
        ]

        # compute residual for rendezvous constraint
        residuals = np.array(
            [
                x_f_nd_p - x_f_target_nd,  # Final x nd
                y_f_nd_p - y_f_target_nd,  # Final y nd
                vx_f_nd_p - vx_f_target_nd,  # Final x velocity constraint
                vy_f_nd_p - vy_f_target_nd,  # Final y velocity constraint
                lam_m_f_nd_p - lam_m_f  # Final mass co-state should be 0
            ]
        )

        self.shooting_iters = self.shooting_iters + 1

        return residuals
    
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
                raise FirstGuessException("Cannot find good initial co-state guess")

            # randomize the first guess if the first guess is no good
            if counter_first_guess > 1:
                lam_guess = rng.normal(
                    loc=mean_co_state_guess,
                    scale=std_co_state_guess,
                    size=len_co_state_guess,
                )

                # add bias array
                lam_guess = lam_guess + bias_co_states

                        # Call Levenberg-Marquardt root finder
            lam_sol = root(
                self.shooting_iteration,
                lam_guess,
                args=(self.eps_0,),
                method="lm",
                options={"ftol": self.root_tol, "maxiter": self.root_max_iters},
            )

            # Calculate actual residual norm and success flag
            residual_norm = np.linalg.norm(lam_sol.fun)
            success = lam_sol.success
            iters_taken = lam_sol.nfev

            if ( success and residual_norm < self.root_tol ):

                self._log_controller_info(
                    "Attempt "
                    + str(counter_first_guess)
                    + "   Lambda: "
                    + str(lam_guess)
                    + " passed"
                )
                self._log_controller_info(
                    f"Initial guess root finding converged with residual: {residual_norm:.4e}"
                )
                self._log_controller_info("f evals: " + " " + str(iters_taken))
                return lam_guess
            
            else:
                self._log_controller_info(
                    "Lambda: "
                    + str(counter_first_guess)
                    + "   lam_guess "
                    + str(lam_guess)
                    + " failed"
                )
                self._log_controller_info(
                    f"Initial guess root finding failed with residual: {residual_norm:.4e}"
                )
                self._log_controller_info("Shooting iters: " + str(self.shooting_iters))
