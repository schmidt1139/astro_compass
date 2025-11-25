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

def convert_fpa_to_velocity_components(v, fpa):
    v_r = v * np.sin(fpa)
    v_theta = v * np.cos(fpa)

    return v_r, v_theta

def convert_radial_velocity_to_cartesian(v_r, v_theta, theta):
    vx = v_r * np.cos(theta) - v_theta * np.sin(theta)
    vy = v_r * np.sin(theta) + v_theta * np.cos(theta)

    return vx, vy

def convert_alpha_from_cart_to_fpa(x, y, vx, vy, alpha_x, alpha_y):
    r, theta, v_r, v_theta = cartesian_to_polar(x, y, vx, vy)

    # rotate alpha components to radial/tangential frame
    alpha_r = alpha_x * np.cos(theta) + alpha_y * np.sin(theta)
    alpha_theta = -alpha_x * np.sin(theta) + alpha_y * np.cos(theta)

    # convert to fpa frame
    alpha_fpa_cos = (v_r * alpha_r + v_theta * alpha_theta) / np.sqrt(v_r**2 + v_theta**2)
    alpha_fpa_sin = (v_theta * alpha_r - v_r * alpha_theta) / np.sqrt(v_r**2 + v_theta**2)

    return alpha_fpa_cos, alpha_fpa_sin

def convert_alpha_from_fpa_to_cart(x, y, vx, vy, alpha_fpa_cos, alpha_fpa_sin):
    r, theta, v_r, v_theta = cartesian_to_polar(x, y, vx, vy)

    # convert to radial/tangential frame
    alpha_r = (v_r * alpha_fpa_cos - v_theta * alpha_fpa_sin) / np.sqrt(v_r**2 + v_theta**2)
    alpha_theta = (v_theta * alpha_fpa_cos + v_r * alpha_fpa_sin) / np.sqrt(v_r**2 + v_theta**2)

    # rotate alpha components to cartesian frame
    alpha_x = alpha_r * np.cos(theta) - alpha_theta * np.sin(theta)
    alpha_y = alpha_r * np.sin(theta) + alpha_theta * np.cos(theta)

    return alpha_x, alpha_y

def convert_attitude_from_radial_to_cartesian(x, y, alpha_radial, alpha_theta):
    
    theta = np.arctan2(y, x)

    # rotate alpha components to cartesian frame
    alpha_x = alpha_radial * np.cos(theta) - alpha_theta * np.sin(theta)
    alpha_y = alpha_radial * np.sin(theta) + alpha_theta * np.cos(theta)

    return alpha_x, alpha_y

def convert_attitude_from_cartesian_to_radial(x, y, alpha_x, alpha_y):
    
    theta = np.arctan2(y, x)

    # rotate alpha components to radial frame
    alpha_radial = alpha_x * np.cos(theta) + alpha_y * np.sin(theta)
    alpha_theta = -alpha_x * np.sin(theta) + alpha_y * np.cos(theta)

    return alpha_radial, alpha_theta