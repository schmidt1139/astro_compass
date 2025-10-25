import numpy as np


def polar_to_cartesian(r, theta, v_r, v_theta):
    # cartesian state vector
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    vx = v_r * np.cos(theta) - v_theta * np.cos(np.pi / 2 - theta)
    vy = v_r * np.sin(theta) + v_theta * np.sin(np.pi / 2 - theta)

    return x, y, vx, vy


def cartesian_to_polar(x, y, vx, vy):
    r = (x**2 + y**2) ** 0.5
    theta = np.atan2(y, x)
    v_r = (x * vx + y * vy) / r
    v_theta = (x * vy - y * vx) / r

    return r, theta, v_r, v_theta

def calc_cart_from_OE( a, e, w, theta, mu_cb):
    
    # calculate radius based off of orbital elements
    r = a * (1 - e**2) / (1 + e * np.cos(theta))

    # calculate velocity components
    h = (mu_cb * a * (1 - e**2)) ** 0.5
    r_dot = (mu_cb / h) * e * np.sin(theta)
    v_theta = (mu_cb / h) * (1 + e * np.cos(theta))

    # convert to cartesian coordinates
    x, y, vx, vy = polar_to_cartesian(r, theta, r_dot, v_theta)

    # rotate by coordinats by argument of periapsis
    x_rot = x * np.cos(w) + y * np.sin(w)
    y_rot = -x * np.sin(w) + y * np.cos(w)
    vx_rot = vx * np.cos(w) + vy * np.sin(w)
    vy_rot = -vx * np.sin(w) + vy * np.cos(w)

    return x_rot, y_rot, vx_rot, vy_rot


def non_dimensionalize(arr_y, g0, mu, T_max, ISP, TOF, l_star, m_star, t_star):
    # unpack the state vector
    x, y, vx, vy, m = arr_y[:5]

    # initialize non-dim variables to input state vector
    x_nd, y_nd, vx_nd, vy_nd, m_nd = x, y, vx, vy, m

    # initial parameters
    g0_nd = g0
    mu_nd = mu
    T_max_nd = T_max
    ISP_nd = ISP
    TOF_nd = TOF

    # non-dimensionalize by length
    x_nd = x_nd / l_star
    y_nd = y_nd / l_star
    vx_nd = vx_nd / l_star
    vy_nd = vy_nd / l_star
    g0_nd = g0_nd / l_star
    mu_nd = mu_nd / (l_star) ** 3
    T_max_nd = T_max_nd / l_star

    # non-dimensionalize by time
    vx_nd = vx_nd * t_star
    vy_nd = vy_nd * t_star
    g0_nd = g0_nd * t_star**2
    mu_nd = mu_nd * t_star**2
    T_max_nd = T_max_nd * t_star**2
    ISP_nd = ISP_nd / t_star
    TOF_nd = TOF_nd / t_star

    # non-dimensionalize by mass
    m_nd = m_nd / m_star
    T_max_nd = T_max_nd / m_star

    # pack nd array
    arr_y_nd = np.array([x_nd, y_nd, vx_nd, vy_nd, m_nd])

    # return outputs
    return arr_y_nd, g0_nd, mu_nd, T_max_nd, ISP_nd, TOF_nd
