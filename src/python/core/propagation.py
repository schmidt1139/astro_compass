import numpy as np
from constants.constants import Constants
from core.exceptions import SpacecraftCollisionException, LowMassException


def spacecraft_EOM_radial_2D_EB(t, y, params):
    """
    ode propagation function
    -----------------------------------------------------------------------------------
    This is a set of equations of motion that govern a spacecraft in the
    2-dimensional problem using the polar form.

    Inputs
    -----------------------------------------------------------------------------------
    t:      Elapsed time
    y:      Input state vector (r,theta,r_dot,v_theta)
    params: The list of parameters

    Outputs
    -----------------------------------------------------------------------------------
    Derivatives for state vector dy
    """

    # get the set of parameters
    params = params.astype(np.float32)
    num_params = len(params)

    # check parameter length
    if num_params != 6:
        raise Exception("Invalid number of parameters")

    # unpack the parameters
    mu_cb = params[0]  # Central body gravitational parameter (km^3/s^2)
    radius_cb = params[1]  # Central body radius (km)
    C1 = params[2]  # Maximum spacecraft thrust
    C2 = params[3]  # Specific impulse
    u = params[4]  # Spacecraft throttle input (action/control)
    beta = params[5]  # Spacecraft thrust angle
    g0 = 9.80665  # Gravity at sea level (m/s^2)

    # unpack state vector
    r = y[0]  # radius
    r_dot = y[2]  # r_dot
    v_theta = y[3]  # v_theta
    m = y[4]  # mass

    # calc derivatives
    d_r = r_dot
    d_theta = v_theta / r
    d_r_dot = v_theta**2 / r - mu_cb / r**2 + C1 / m * u * np.sin(beta)
    d_v_theta = -r_dot * v_theta / r + C1 / m * u * np.cos(beta)
    d_m = -C1 * u / g0 / C2  # kg/s

    # construct derivative array
    dy = np.zeros(5, dtype=np.float32)

    # assign derivatives if the spacecraft has not impacted central body,
    # otherwise keep derivatives at zero
    if r > radius_cb:
        dy[0] = d_r
        dy[1] = d_theta
        dy[2] = d_r_dot
        dy[3] = d_v_theta
        dy[4] = d_m

    # return derivative array
    return dy.astype(np.float32)


# method for calculating spacecraft EOM for Hohmann transfer env
def spacecraft_EOM_f_2D_2B(self, t, y, params):
    """
    ode propagation function
    -----------------------------------------------------------------------------------
    This is a special case force model function for use in propagating a
    spacecraft. This function assumes that there are only two dimensions
    (x and y) and that there are only two bodies: the spacecraft and the
    central body.

    Inputs
    -----------------------------------------------------------------------------------
    t:      Elapsed time
    y:      Input state vector
    params: The list of parameters

    Outputs
    -----------------------------------------------------------------------------------
    """

    dy = np.zeros(4, dtype=np.float32)
    params = params.astype(np.float32)

    num_params = len(params)

    if num_params == 4:
        mu_cb = params[0]
        radius_cb = params[1]
        x_cb = params[2]
        y_cb = params[3]
    else:
        raise Exception("Invalid number of parameters")

    # unpack the state vector
    x_sc = y[0]
    y_sc = y[1]
    vx_sc = y[2]
    vy_sc = y[3]

    # determine relative position
    x_rel = x_sc - x_cb
    y_rel = y_sc - y_cb

    # relative position magnitude
    r_rel = (x_rel**2 + y_rel**2) ** (0.5)

    if r_rel < radius_cb:
        dy[0] = 0.0
        dy[1] = 0.0
        dy[2] = 0.0
        dy[3] = 0.0

        return dy.astype(np.float32)

    else:
        x_hat_rel = x_rel / r_rel
        y_hat_rel = y_rel / r_rel

        ddx = -mu_cb * x_hat_rel / r_rel**2
        ddy = -mu_cb * y_hat_rel / r_rel**2

        dy[0] = vx_sc
        dy[1] = vy_sc
        dy[2] = ddx
        dy[3] = ddy

        return dy.astype(np.float32)


def Hamiltonian_EOM_TBT(t, y, params):
    """
    Two-Body Orbit Transfer Hamiltonian ode propagation function
    -----------------------------------------------------------------------------------
    This is a set of equations of motion that govern a spacecraft in the
    2-dimensional problem using the polar form while performing optimal control
    using a Hamiltonian control algorithm. The state vector consists of a
    spacecraft's planar state vector along with the associated co-states that
    must be propagated with the state variables as well.

    Inputs
    -----------------------------------------------------------------------------------
    t:      Elapsed time
    y:      Input state vector (r,theta,r_dot,v_theta,m,lambda_r,
                                lambda_theta,lambda_r_dot,lambda_v_theta,
                                lambda_m)
    params: The list of parameters

    Outputs
    -----------------------------------------------------------------------------------
    Derivatives for state vector dy
    """

    # Scale factors for non-physical parameters
    scale_factor_lambda_m = 1

    # get the set of parameters
    params = params.astype(np.float32)
    num_params = len(params)

    # check parameter length
    if num_params != 3:
        raise Exception("Invalid number of parameters")

    # unpack the parameters
    mu_cb = params[0]  # Central body gravitational parameter (m^3/s^2)
    C1 = params[1]  # Maximum spacecraft thrust in N
    C2 = params[2]  # Specific impulse (s)
    g0 = 9.80665  # Gravity at sea level (m/s^2)

    # unpack the state vector
    r, theta, r_dot, v_theta, m = y[:5]
    lambda_r, lambda_theta, lambda_r_dot, lambda_v_theta, lambda_m = y[5:10]

    # scale variables as needed
    lambda_m = lambda_m * scale_factor_lambda_m

    # calculate the optimal control actions
    beta = np.atan2(lambda_r_dot, lambda_v_theta)
    u = -lambda_m * C1 / C2
    u = u + C1 * (lambda_r_dot * np.sin(beta) + lambda_v_theta * np.cos(beta)) / m
    u = np.clip(u, 0, 1)

    # state vector EOM
    d_r = r_dot  # km
    d_theta = v_theta / r  # rad
    d_r_dot = v_theta**2 / r - mu_cb / r**2 + C1 / m * u * np.sin(beta)  # km/s
    d_v_theta = -r_dot * v_theta / r + C1 / m * u * np.cos(beta)  # km/s
    d_m = -C1 * u / g0 / C2  # kg/s

    # co-state vector EOM
    d_lambda_r = -2 * lambda_r_dot * mu_cb / r**3 + lambda_r_dot * v_theta**2 / r**2
    d_lambda_r = d_lambda_r - lambda_v_theta * r_dot * v_theta / r**2

    d_lambda_theta = 0

    d_lambda_r_dot = -lambda_r + lambda_v_theta * v_theta / r
    d_lambda_v_theta = (
        -lambda_theta - 2 * lambda_r_dot * v_theta / r + lambda_v_theta * r_dot / r
    )
    d_lambda_m = (
        lambda_r_dot * C1 * u * np.sin(beta) + lambda_v_theta * C1 * u * np.cos(beta)
    ) / m**2

    # initialize derivative vector
    dy = np.zeros(10, dtype=np.float32)

    # assign derivatives to output vector
    dy[0] = d_r
    dy[1] = d_theta
    dy[2] = d_r_dot
    dy[3] = d_v_theta
    dy[4] = d_m
    dy[5] = d_lambda_r
    dy[6] = d_lambda_theta
    dy[7] = d_lambda_r_dot
    dy[8] = d_lambda_v_theta
    dy[9] = d_lambda_m

    # error handling
    if d_m > 0:
        raise Exception("Error: Positive mass rate detected")

    return dy.astype(np.float32)


def Hamiltonian_EOM_TBT_nd(t, y, params):
    """
    Two-Body Orbit Transfer Hamiltonian Non-Dimensional ode propagation
    function.
    -----------------------------------------------------------------------------------
    This is a set of equations of motion that govern a spacecraft in the
    2-dimensional problem using the polar form while performing optimal control
    using a Hamiltonian control algorithm. The state vector consists of a
    spacecraft's planar state vector along with the associated co-states that
    must be propagated with the state variables as well. All of the parameters
    in this implementation are assumed to be non-dimensional.

    Inputs
    -----------------------------------------------------------------------------------
    t:      Elapsed time
    y:      Input state vector (r,theta,r_dot,v_theta,m,lambda_r,
                                lambda_theta,lambda_r_dot,lambda_v_theta,
                                lambda_m)
    params: The list of parameters

    Outputs
    -----------------------------------------------------------------------------------
    Derivatives for state vector dy
    """

    # get the set of parameters
    params = params.astype(np.float32)
    num_params = len(params)

    # check parameter length
    if num_params != 6:
        raise Exception("Invalid number of parameters")

    # unpack the parameters
    mu_cb = params[0]  # Central body gravitational parameter
    C1 = params[1]  # Maximum spacecraft thrust (N)
    C2 = params[2]  # Specific impulse (s)
    l_star = params[3]  # Characteristic length (m)
    m_star = params[4]  # Characteristic mass (kg)
    t_star = params[5]  # Characteristic time (s)

    g0 = 9.80665  # Acceleration at Earth surface (m/s^2)

    # unpack the state vector
    r_nd, theta_nd, r_dot_nd, v_theta_nd, m_nd = y[:5]
    lambda_r_nd, lambda_theta_nd, lambda_r_dot_nd, lambda_v_theta_nd, lambda_m_nd = y[
        5:10
    ]

    # scale state vector
    r = r_nd * l_star
    r_dot = r_dot_nd * l_star / t_star
    v_theta = v_theta_nd * l_star / t_star
    m = m_nd * m_star

    # scale co-states
    lambda_r = lambda_r_nd * l_star
    lambda_theta = lambda_theta_nd
    lambda_r_dot = lambda_r_dot_nd * l_star / t_star
    lambda_v_theta = lambda_v_theta_nd * l_star / t_star
    lambda_m = lambda_m_nd * m_star

    # calculate the optimal control actions
    beta = np.atan2(lambda_r_dot, lambda_v_theta)
    u = -lambda_m * C1 / C2
    u = u + C1 * (lambda_r_dot * np.sin(beta) + lambda_v_theta * np.cos(beta)) / m
    u = np.clip(u, 0, 1)

    # state vector EOM
    d_r = r_dot
    d_theta = v_theta / r
    d_r_dot = v_theta**2 / r - mu_cb / r**2 + C1 / m * u * np.sin(beta)
    d_v_theta = -r_dot * v_theta / r + C1 / m * u * np.cos(beta)
    d_m = -C1 * u / C2 / g0

    # co-state vector EOM
    d_lambda_r = -2 * lambda_r_dot * mu_cb / r**3 + lambda_r_dot * v_theta**2 / r**2
    d_lambda_r = d_lambda_r - lambda_v_theta * r_dot * v_theta / r**2

    d_lambda_theta = 0

    d_lambda_r_dot = -lambda_r + lambda_v_theta * v_theta / r
    d_lambda_v_theta = (
        -lambda_theta - 2 * lambda_r_dot * v_theta / r + lambda_v_theta * r_dot / r
    )
    d_lambda_m = (
        lambda_r_dot * C1 * u * np.sin(beta) + lambda_v_theta * C1 * u * np.cos(beta)
    ) / m**2

    # initialize derivative vector
    dy = np.zeros(10, dtype=np.float32)

    # scale state vector for output
    d_r_nd = d_r / l_star
    d_theta_nd = d_theta
    d_r_dot_nd = d_r_dot * t_star / l_star
    d_v_theta_nd = d_v_theta * t_star / l_star
    d_m_nd = d_m / m_star

    # scale co-state vector
    d_lambda_r_nd = d_lambda_r / l_star
    d_lambda_theta_nd = d_lambda_theta
    d_lambda_r_dot_nd = d_lambda_r_dot * t_star / l_star
    d_lambda_v_theta_nd = d_lambda_v_theta * t_star / l_star
    d_lambda_m_nd = d_lambda_m / m_star

    # assign derivatives to output vector
    dy[0] = d_r_nd
    dy[1] = d_theta_nd
    dy[2] = d_r_dot_nd
    dy[3] = d_v_theta_nd
    dy[4] = d_m_nd
    dy[5] = d_lambda_r_nd
    dy[6] = d_lambda_theta_nd
    dy[7] = d_lambda_r_dot_nd
    dy[8] = d_lambda_v_theta_nd
    dy[9] = d_lambda_m_nd

    dy = dy

    # error handling
    if d_m > 0:
        raise Exception("Error: Positive mass rate detected")

    return dy.astype(np.float32)


def Hamiltonian_EOM_TBT_v2(t, state, params):
    """
    Two-Body Orbit Transfer Hamiltonian ode propagation function
    -----------------------------------------------------------------------------------
    This is a set of equations of motion that govern a spacecraft in the
    2-dimensional problem using the polar form while performing optimal control
    using a Hamiltonian control algorithm. The state vector consists of a
    spacecraft's planar state vector along with the associated co-states that
    must be propagated with the state variables as well.

    Inputs
    -----------------------------------------------------------------------------------
    t:      Elapsed time
    y:      Input state vector (x,y,vx,vy,m,lambda_x, lambda_y,lambda_vx,
                                lambda_vy, lambda_m)
    params: The list of parameters

    Outputs
    -----------------------------------------------------------------------------------
    Derivatives for state vector dy
    """

    # get the number of parameters
    num_params = len(params)

    max_deriv = 10000.0

    # check parameter length
    if num_params != 10:
        raise Exception("Invalid number of parameters")

    # unpack the parameters
    mu = params[0]  # gravitational parameter of the central body
    T_max = params[1]  # max thrust of the spacecraft
    ISP = params[2]  # spacecraft specific impulse
    g0 = params[6]  # acceleration at Earth surface
    eps = params[7]  # smoothing parameter
    flag_constrain_u = params[8]  # Boolean indicating if 0<u<1 is enforced
    switch_smoothing_method = params[9]  # Smoothing method

    # unpack the state vector
    x, y, vx, vy, m = state[:5]

    # unpack the co-state vector
    lam_x, lam_y, lam_vx, lam_vy, lam_m = state[5:10]

    # create vectors
    r_vec = np.array([x, y])
    v_vec = np.array([vx, vy])
    lam_r_vec = np.array([lam_x, lam_y])
    lam_v_vec = np.array([lam_vx, lam_vy])

    # Derivative calculation preliminaries
    r = np.linalg.norm(r_vec)

    # Check if spacecraft is too close to central body (prevents numerical issues)
    r_min = 0.01  # Minimum allowed radius (non-dimensional)
    if r < r_min:
        raise SpacecraftCollisionException(
            f"Spacecraft too close to central body: r = {r:.6e}"
        )

    r_dot_lam_v = np.dot(r_vec, lam_v_vec)
    lam_v_mag = np.linalg.norm(lam_v_vec)

    # Determine control--------------------------------------------------------
    # Alpha vector is opposite of ideal thrust direction - lambda v unit vector
    alpha_vec = -lam_v_vec / lam_v_mag

    # Switching function
    rho = lam_m + ISP * g0 * lam_v_mag / m - 1

    # Determine u based on switching function and smoothing parameter. If the
    # smoothing parameter is exactly zero, then no smoothing is to be performed
    # on the throttle u. If eps is non-zero, then smoothing is to be applied
    # to u according to the smoothing method used.

    if eps == 0.0:
        if rho >= 0:
            u = 1.0
        else:
            u = 0.0
    else:
        # check the smoothing method
        if switch_smoothing_method == 0:
            u = smoothing_function_tanh(rho, eps)
        elif switch_smoothing_method == 1:
            u = smoothing_function_homotopic(rho, eps, flag_constrain_u)

    # Check if spacecraft mass is too low (can cause numerical issues)
    m_min = 0.01  # Minimum allowed mass (non-dimensional) - 1% of initial mass
    if m < m_min:
        u = 0.0  # No thrust if out of fuel

    # state vector derivative calculations
    dr_vec = v_vec
    dv_vec = -mu / r**3 * r_vec + u * T_max / m * alpha_vec
    dm = -T_max * u / ISP / g0

    if dm > 0.0:
        raise Exception("Positive mass rate detected")

    # In Hamiltonian_EOM_TBT_v2
    r = np.linalg.norm(r_vec)
    r_min = 0.001  # Minimum allowed radius (non-dimensional)
    if r < r_min:
        raise Exception(f"Spacecraft too close to central body: r = {r}")

    # co-state vector derivatives
    d_lam_r_vec = mu / r**3 * lam_v_vec - 3 * mu * r_dot_lam_v / r**5 * r_vec
    d_lam_v_vec = -lam_r_vec
    d_lam_m = -T_max * u / m**2 * lam_v_mag

    # pack derivatives into output array
    derivs = np.array(
        [
            dr_vec[0],
            dr_vec[1],
            dv_vec[0],
            dv_vec[1],
            dm,
            d_lam_r_vec[0],
            d_lam_r_vec[1],
            d_lam_v_vec[0],
            d_lam_v_vec[1],
            d_lam_m,
        ]
    )

    return derivs


def smoothing_function_tanh(rho, eps):
    u_smooth = 1 / 2 * (1 + np.tanh(rho / eps))

    return u_smooth


def smoothing_function_homotopic(rho, eps, flag_constrain_u):
    # If u is unconstrained
    if not flag_constrain_u:
        u_smooth = 1 / 2 + (rho / 2 * eps)

    # Throttle input is bounded between 0 and 1
    else:
        if rho > eps:
            u_smooth = 1
        elif rho < -eps:
            u_smooth = 0
        else:
            u_smooth = 1 / 2 + (rho / 2 * eps)

    return u_smooth


def env_EOM_TBT_v2(t, state, params):
    """
    Two-Body Orbit Transfer Gym Environment ode propagation function
    -----------------------------------------------------------------------------------
    This is a set of equations of motion that govern a spacecraft in the
    2-dimensional two-body problem, formatted specifically to the Two-Body
    Transfer gym environment

    Inputs
    -----------------------------------------------------------------------------------
    t:      Elapsed time
    state:  Input state vector (x,y,vx,vy,m)
    params: The list of parameters ()

    Outputs
    -----------------------------------------------------------------------------------
    Derivatives for state vector dy
    """

    # get the number of parameters
    num_params = len(params)

    # check parameter length
    if num_params != 7:
        raise Exception("Invalid number of parameters")

    # unpack the state vector
    x, y, vx, vy, m = state[:5]

    # check non dim components
    t_star = (149598023000**3 / (Constants.MU_SUN * 10 ** (9))) ** 0.5
    x_nd = x / Constants.SMA_EARTH
    y_nd = y / Constants.SMA_EARTH
    vx_nd = vx / Constants.SMA_EARTH * t_star
    vy_nd = vy / Constants.SMA_EARTH * t_star
    m_nd = m / 3366.0

    # unpack the parameters
    mu = params[0]  # gravitational parameter of the central body
    T_max = params[1]  # max thrust of the spacecraft
    ISP = params[2]  # spacecraft specific impulse
    g0 = params[3]  # acceleration at Earth surface
    u = params[4]  # spacecraft throttle control
    alpha_x = params[5]  # thrust x direction
    alpha_y = params[6]  # thrust y direction

    # create vectors
    r_vec = np.array([x, y])
    v_vec = np.array([vx, vy])
    alpha_vec = np.array([alpha_x, alpha_y])

    a_vec_mag = (alpha_x**2 + alpha_y**2) ** 0.5

    # enforce unit vector
    if a_vec_mag >= 0.000001:
        alpha_vec = alpha_vec / a_vec_mag
    else:
        if u > 0:
            raise Exception(
                "Error: Zero (or-near zero) thrust direction vector detected"
            )

    # Derivative calculation preliminaries
    r = np.linalg.norm(r_vec)

    # state vector derivative calculations
    dr_vec = v_vec
    dv_vec = -mu / r**3 * r_vec + u * T_max / m * alpha_vec
    dm = -T_max * u / ISP / g0

    derivs = np.array(
        [
            dr_vec[0],
            dr_vec[1],
            dv_vec[0],
            dv_vec[1],
            dm,
        ]
    )

    return derivs
