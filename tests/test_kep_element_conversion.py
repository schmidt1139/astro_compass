from utils.state_vector_utils import calc_cart_from_OE, calc_OE_from_cart
from constants.constants import Constants
from utils.log_utils import log
import numpy as np

def test_kep_element_conversion(flag_report_live):

    test_log = []

    if flag_report_live:
        print("\nRunning test_kep_element_conversion...\n")

    a_nd = 1.5 # semi-major axis in non-dimensional units
    e = 0.1    # eccentricity
    w_deg = 45.0  # argument of periapsis in degrees
    theta_deg = 30.0  # true anomaly in degrees

    eps = 1e-6  # tolerance for floating-point comparisons

    a_m = a_nd * Constants.SMA_EARTH  # convert semi-major axis to meters
    w_rad = np.deg2rad(w_deg)
    theta_rad = np.deg2rad(theta_deg)

    test_log = log(f"Input a: {a_m} m", test_log, flag_report_live)
    test_log = log(f"Input e: {e}", test_log, flag_report_live)
    test_log = log(f"Input w: {w_rad} rad", test_log, flag_report_live)
    test_log = log(f"Input theta: {theta_rad} rad", test_log, flag_report_live)

    # convert to cartesian state vector
    x, y, vx, vy = calc_cart_from_OE(a_m, e, w_rad, theta_rad, Constants.MU_SUN_M )

    test_log = log(f"Converted to cartesian state vector:", test_log, flag_report_live)
    test_log = log(f"x: {x} m", test_log, flag_report_live)
    test_log = log(f"y: {y} m", test_log, flag_report_live)
    test_log = log(f"vx: {vx} m/s", test_log, flag_report_live)
    test_log = log(f"vy: {vy} m/s", test_log, flag_report_live)

    # convert back to Keplerian elements
    a_conv, e_conv, w_conv_rad, theta_conv_rad = calc_OE_from_cart(x, y, vx, vy, Constants.MU_SUN_M )

    test_log = log(f"Converted back to Keplerian elements:", test_log, flag_report_live)
    test_log = log(f"a: {a_conv} m", test_log, flag_report_live)
    test_log = log(f"e: {e_conv}", test_log, flag_report_live)
    test_log = log(f"w: {w_conv_rad} rad", test_log, flag_report_live)
    test_log = log(f"theta: {theta_conv_rad} rad", test_log, flag_report_live)

    a_conv_nd = a_conv / Constants.SMA_EARTH
    w_conv_deg = np.rad2deg(w_conv_rad)
    theta_conv_deg = np.rad2deg(theta_conv_rad)

    a_diff = abs(a_conv_nd - a_nd)
    e_diff = abs(e_conv - e)
    w_diff = abs(w_conv_deg - w_deg)
    theta_diff = abs(theta_conv_deg - theta_deg)

    test_log = log(f"Converted a back to non-dimensional: {a_conv_nd}", test_log, flag_report_live)
    test_log = log(f"Converted e back: {e_conv}", test_log, flag_report_live)
    test_log = log(f"Converted w back to degrees: {w_conv_deg}", test_log, flag_report_live)
    test_log = log(f"Converted theta back to degrees: {theta_conv_deg}", test_log, flag_report_live)

    flag_pass = True
    if a_diff < eps:
        test_log = log("Semi-major axis conversion PASSED", test_log, flag_report_live)
    else:
        flag_pass = False
        test_log = log("Semi-major axis conversion FAILED", test_log, flag_report_live)
    
    if e_diff < eps:
        test_log = log("Eccentricity conversion PASSED", test_log, flag_report_live)
    else:
        flag_pass = False
        test_log = log("Eccentricity conversion FAILED", test_log, flag_report_live)
    if w_diff < eps:
        test_log = log("Argument of periapsis conversion PASSED", test_log, flag_report_live)
    else:
        flag_pass = False
        test_log = log("Argument of periapsis conversion FAILED", test_log, flag_report_live)
    if theta_diff < eps:
        test_log = log("True anomaly conversion PASSED", test_log, flag_report_live)
    else:
        flag_pass = False
        test_log = log("True anomaly conversion FAILED", test_log, flag_report_live)

    if (flag_pass):
        test_log = log("\nAll Keplerian element conversions PASSED\n", test_log, flag_report_live)
    else:
        test_log = log("\nSome Keplerian element conversions FAILED\n", test_log, flag_report_live)
    
    return flag_pass

if __name__ == "__main__":
    test_kep_element_conversion(True)  # Set to True for verbose output