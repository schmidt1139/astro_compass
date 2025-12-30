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
    
    # Safe handling of r=0
    if np.isscalar(r):
        if r < 1e-10:  # Treat near-zero as zero
            v_r = 0.0
            v_theta = 0.0
        else:
            v_r = (x * vx + y * vy) / r
            v_theta = (x * vy - y * vx) / r
    else:
        v_r = np.zeros_like(r)
        v_theta = np.zeros_like(r)
        mask = r >= 1e-10
        v_r[mask] = (x[mask] * vx[mask] + y[mask] * vy[mask]) / r[mask]
        v_theta[mask] = (x[mask] * vy[mask] - y[mask] * vx[mask]) / r[mask]

    return r, theta, v_r, v_theta


def calc_cart_from_OE(a, e, w, theta, mu_cb):
    # calculate radius based off of orbital elements
    r = a * (1 - e**2) / (1 + e * np.cos(theta))

    # calculate velocity components
    h = (mu_cb * a * (1 - e**2)) ** 0.5
    r_dot = (mu_cb / h) * e * np.sin(theta)
    v_theta = (mu_cb / h) * (1 + e * np.cos(theta))

    # convert to cartesian coordinates
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)
    vx_orb = r_dot * np.cos(theta) - v_theta * np.sin(theta)
    vy_orb = r_dot * np.sin(theta) + v_theta * np.cos(theta)

    # rotate by coordinats by argument of periapsis
    cos_w = np.cos(w)
    sin_w = np.sin(w)

    x_rot = x_orb * cos_w - y_orb * sin_w
    y_rot = x_orb * sin_w + y_orb * cos_w
    vx_rot = vx_orb * cos_w - vy_orb * sin_w
    vy_rot = vx_orb * sin_w + vy_orb * cos_w

    return x_rot, y_rot, vx_rot, vy_rot

def calc_OE_from_cart(x, y, vx, vy, mu_cb):
        
    # determine coordinates relative to central body
    x_rel_km = x / 1000
    y_rel_km = y / 1000

    # position and velocity magnitudes
    r_km = (x_rel_km**2 + y_rel_km**2) ** 0.5
    r = r_km * 1000  # convert back to m

    flag_non_kep = False
    
    # Safe handling of r=0
    if r < 1e-10:
        print(f"Impact detected at position: x={x}, y={y}")
        print(f"Current state vector: x={x}, y={y}, vx={vx}, vy={vy}")
        flag_non_kep = True
        #raise ValueError("Position radius is zero or near-zero. Cannot calculate orbital elements.")

    # spacecraft position and velocity vectors
    sc_pos = np.array([x, y, 0.0])
    sc_vel = np.array([vx, vy, 0.0])
    r_hat = sc_pos / r

    # angular momentum
    h_vec = np.cross(sc_pos, sc_vel)
    h = np.linalg.norm(h_vec)
    
    # Safe handling of h=0 (degenerate orbit)
    if h < 1e-10:
        print(f"Degenerate orbit detected at position: x={x}, y={y}")
        print(f"Current state vector: x={x}, y={y}, vx={vx}, vy={vy}")
        flag_non_kep = True
        #raise ValueError("Angular momentum is zero or near-zero. Orbit is degenerate (radial motion only).")
    
    if not flag_non_kep:
        h_hat = h_vec / h

        # eccentricity vector
        e_vec = np.cross(sc_vel, h_vec) / mu_cb - sc_pos / r
        e = np.linalg.norm(e_vec)

        # If e is zero, we will get an error dividing by zero, so the ecc vector
        # is set at {0,0,0} if the magnitude is zero.
        if e == 0.0:
            e_hat = e_vec * 0.0
        else:
            e_hat = e_vec / e

        # semi major axis
        rp = h**2 / mu_cb / (1 + e * np.cos(0))
        ra = h**2 / mu_cb / (1 + e * np.cos(np.pi))
        a = 1 / 2 * (rp + ra)

        # argument of periapsis - for planar orbit, angle from x-axis to eccentricity vector
        w = np.arctan2(e_vec[1], e_vec[0])
        if w < 0:
            w = w + 2 * np.pi

        # true anomaly - extra error handling included,
        # mainly needed for hyperbolic instances
        # Compute true anomaly using atan2 for proper quadrant handling
        r_vec = np.array([x, y, 0.0])
        theta = np.arctan2(np.dot(h_hat, np.cross(e_hat, r_vec)), np.dot(e_hat, r_vec))
        if theta < 0:
            theta = theta + 2 * np.pi
    else:
        a = 0.0
        e = 0.0
        w = 0.0
        theta = 0.0

    return a, e, w, theta


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

    # Safe handling of v=0 (stationary object)
    v_mag = np.sqrt(v_r**2 + v_theta**2)
    if v_mag < 1e-10:
        raise ValueError("Velocity magnitude is zero or near-zero. Cannot convert to FPA frame.")

    # convert to fpa frame
    alpha_fpa_cos = (v_r * alpha_r + v_theta * alpha_theta) / v_mag
    alpha_fpa_sin = (v_theta * alpha_r - v_r * alpha_theta) / v_mag

    return alpha_fpa_cos, alpha_fpa_sin


def convert_alpha_from_fpa_to_cart(x, y, vx, vy, alpha_fpa_cos, alpha_fpa_sin):
    r, theta, v_r, v_theta = cartesian_to_polar(x, y, vx, vy)

    # Safe handling of v=0 (stationary object)
    v_mag = np.sqrt(v_r**2 + v_theta**2)
    if v_mag < 1e-10:
        raise ValueError("Velocity magnitude is zero or near-zero. Cannot convert from FPA frame.")

    # convert to radial/tangential frame
    alpha_r = (v_r * alpha_fpa_cos - v_theta * alpha_fpa_sin) / v_mag
    alpha_theta = (v_theta * alpha_fpa_cos + v_r * alpha_fpa_sin) / v_mag

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

def orbit_equation(theta, a, e):
    # calculate radius based off of orbital elements
    r = a * (1 - e**2) / (1 + e * np.cos(theta))

    return r

def compute_kep_velocities(a, e, theta, mu_cb):

    # calculate radius based off of orbital elements
    r = a * (1 - e**2) / (1 + e * np.cos(theta))

    # compute semilatus rectum
    p = a * (1 - e**2)

    # compute angular momentum
    h = (mu_cb * p) ** 0.5

    # calculate velocity components
    r_dot = (mu_cb / h) * e * np.sin(theta)
    v_theta = (mu_cb / h) * (1 + e * np.cos(theta))

    # total velocity
    v_tot = np.sqrt(r_dot**2 + v_theta**2)

    return r_dot, v_theta, v_tot

def compute_eccentric_anomaly_from_true_anomaly(theta, e):
    # compute eccentric anomaly from true anomaly
    if e < 1:
        E = 2 * np.arctan( np.tan( theta / 2 ) * ( (1 - e) / (1 + e) )**0.5 )
    else:
        E = np.nan

    # ensure E is in the range 0 to 2pi
    if E < 0:
        E = E + 2 * np.pi

    return E

def compute_mean_anomaly_from_eccentric_anomaly(E, e):
    # compute mean anomaly from eccentric anomaly
    M = E - e * np.sin(E)

    return M

