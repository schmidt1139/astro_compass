
import numpy as np;
import gymnasium as gym;
import matplotlib.pyplot as plot;
import matplotlib.ticker as ticker
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register
from stable_baselines3 import A2C

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

from Ephemeris import Ephemeris
from Hamiltonian_Control import Hamiltonian_Controller_TBT
from scipy.integrate import solve_ivp
from Propagation import Hamiltonian_EOM_TBT_v2
from StateVectorUtilities import non_dimensionalize


#register the environment if it isn't registered
if ( ("TwoBody_Orb2Orb_Transfer_Env-v0" in envs.registry.keys()) == False ):
    
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )
    


#initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0");


num_traj = 1;

#The prescribed time of flight for the transfer trajectory [s]
input_TOF = 1.1 * 365.25 * 24 * 60 * 60;
steps_per_traj = np.ceil( input_TOF / env.unwrapped.step_size );

np.set_printoptions(precision=3)  # Limit to 3 decimal places
np.set_printoptions(linewidth=1000)


    
eph = Ephemeris( );
init_observation, init_info = env.reset();
sun_rad = env.unwrapped.planet_radii[0];
T_max = init_info["max_thrust"]*1000; #max thrust in N
ISP = init_info["ISP"];

#compute Hamiltonian Solution
H_controller = Hamiltonian_Controller_TBT(env, init_observation, 
                                          init_info, input_TOF );

mu = env.unwrapped.arr_mu[0] * 1000**3;

#initial state vector
r_0         = 2.32495e8 * 1000;             #m
theta_0     = 0.0;                          #rad
r_dot_0     = 0.0;                          #m/s
v_theta_0   = ( mu / r_0 )**0.5;            #circular vel
m_0         = 3366.0;                       #kg
r_f         = init_observation[6]*1000;     #m
r_dot_f     = 0.0;                          #m/s
v_theta_f   = (mu/r_f) ** 0.5;              #m/s

#constants/non-dim factors
sma_Earth   = 149598023 * 1000;             #m
sma_Mars    = 2.32495e8 * 1000;             #m
g0          = 9.80665;                      #m/s^2
l_star      = sma_Earth;
m_star      = m_0;                          #initial mass in kg
t_star      = (l_star**3/mu)**0.5;          #characteristic time in s
ve          = g0 * ISP;

#cartesian state vector
x0  = r_0 * np.cos(theta_0);
y0  = r_0 * np.sin(theta_0);
vx0 = r_dot_0 * np.cos(theta_0) - v_theta_0 * np.cos(np.pi/2 - theta_0);
vy0 = r_dot_0 * np.sin(theta_0) + v_theta_0 * np.sin(np.pi/2 - theta_0);

#inital lambda guess
lam_x0 = 0.01;
lam_y0 = 0.01;
lam_vx = -0.2;
lam_vy = 0.01;
lam_m = 0.5;
lam_guess = np.array( [lam_x0, lam_y0, lam_vx, lam_vy, lam_m ] );

#initial state vector
arr_y0 =  np.array( [x0, y0, vx0, vy0, m_0 ] );

#non-dim state and parameters
arr_y_nd, g0_nd, mu_nd, T_max_nd, ISP_nd, input_TOF_nd = non_dimensionalize( arr_y0, g0, mu, T_max, ISP, input_TOF, l_star, m_star, t_star );

#construct full state vector at t=0
arr_full_y0 = np.hstack( (arr_y_nd, lam_guess) );

#define the integration time span
t_span = (0,input_TOF_nd);
t_eval = np.linspace(*t_span, 1000);

#set up parameter array
params = np.array( [mu_nd, T_max_nd, ISP_nd, sma_Earth, m_star, t_star, g0_nd ] );

#check initial derivatives
derivs = Hamiltonian_EOM_TBT_v2( 0.0, arr_full_y0, params );

#integrate forward in time
sol = solve_ivp(Hamiltonian_EOM_TBT_v2, t_span, arr_full_y0, method='RK45', args=(params,), rtol=1e-10, atol=1e-14, t_eval=t_eval );

#check if integration was successful
if ( sol.status == -1 ):
    print(sol.message);
    raise Exception("Integration failed");
 
#extract time and state variables
arr_time    = sol.t;
variables   = sol.y;    

arr_x = variables[0];
arr_y = variables[1];
arr_vx = variables[2];
arr_vy = variables[3];
arr_m = variables[4];
arr_lam_x = variables[5];
arr_lam_y = variables[6];
arr_lam_vx = variables[7];
arr_lam_vy = variables[8];
arr_lam_m = variables[9];

arr_E = [];
arr_H = [];
arr_E0 = [];
arr_H0 = [];
arr_m = [];
arr_u = [];
arr_alpha_x = [];
arr_alpha_y = [];
arr_rho = [];
E0 = 1/2 * ((variables[2,0]*l_star / t_star)**2 + (variables[3,0]*l_star / t_star)**2) - mu / r_0;


#ephemeris object initialization
eph = Ephemeris();

#plotting solution output data
for index, t in enumerate(arr_time):
    
    x_i = variables[0,index] * l_star;
    y_i = variables[1,index] * l_star;
    vx_i = variables[2,index] * l_star / t_star;
    vy_i = variables[3,index] * l_star / t_star;
    m_i_nd = variables[4,index];
    lam_x_i = variables[5,index];
    lam_y_i = variables[6,index];
    lam_vx_i = variables[7,index];
    lam_vy_i = variables[8,index];
    lam_m_i = variables[9,index];
    
    r_i = np.linalg.norm([x_i, y_i]);
    r_vec = np.array([x_i, y_i, 0]);
    v_vec = np.array([vx_i, vy_i, 0]);
    lam_v_vec = np.array([lam_vx_i, lam_vy_i]);
    lam_v_mag = np.linalg.norm(lam_v_vec);
    
    #Check orbital energy and angular momentum
    E = 1/2 * (vx_i**2 + vy_i**2) - mu / r_i;
    H_vec = np.cross(r_vec, v_vec);
    H = np.linalg.norm(H_vec);
    
    #Find alpha vector
    alpha_vec = - lam_v_vec / lam_v_mag;
    
    #Switching function
    rho = lam_m_i + ISP_nd * g0_nd * lam_v_mag / m_i_nd - 1;
    
    #Control policy
    if (rho >= 0.0 ):
        u = 1.0;
    else:
        u = 0.0;
    
    #Add data to ephemeris object
    eph.add_data( t, x_i, y_i, vx_i, vy_i );
    
    #append to arrays
    arr_E.append(E);
    arr_m.append(m_i_nd);
    arr_H.append(H);
    arr_E0.append(E0);
    arr_alpha_x.append(alpha_vec[0]);
    arr_alpha_y.append(alpha_vec[1]);
    arr_rho.append(rho);
    arr_u.append(u);
    


fig, ax = plot.subplots(figsize=(6, 6));
ax.plot( arr_time, arr_x, label="x" );
ax.plot( arr_time, arr_y, label="y" );
ax.plot( arr_time, arr_vx, label="vx" );
ax.plot( arr_time, arr_vy, label="vy" );
ax.plot( arr_time, arr_m, label="m" );
ax.set_title("Non-Dim States over time");   
ax.legend();

fig, ax = plot.subplots(figsize=(6, 6));
ax.plot( arr_time, arr_lam_x, label="x" );
ax.plot( arr_time, arr_lam_y, label="y" );
ax.plot( arr_time, arr_lam_vx, label="vx" );
ax.plot( arr_time, arr_lam_vy, label="vy" );
ax.plot( arr_time, arr_lam_m, label="m" );
ax.set_title("Non-Dim Co-States over time");   
ax.legend();

fig, ax = plot.subplots(figsize=(6, 6));
ax.plot( arr_time, arr_E );
ax.plot( arr_time, arr_E0 );
ax.set_title("Orbit Energy over Time");   

fig, ax = plot.subplots(figsize=(6, 6));
ax.plot( arr_time, arr_alpha_x, label="alpha_x" );
ax.plot( arr_time, arr_alpha_y, label="alpha_y" );
ax.set_title("Alpha Vector (Maneuver Direction) over Time"); 
ax.legend();  

fig, ax = plot.subplots(figsize=(6, 6));
ax.plot( arr_time, arr_m );
ax.set_title("Spacecraft Mass Fraction over Time"); 

fig, ax = plot.subplots(figsize=(6, 6));
ax.plot( arr_time, arr_u );
ax.set_title("Spacecraft Thrust Throttle over Time"); 

fig, ax = plot.subplots(figsize=(6, 6));
ax.plot( arr_time, arr_rho );
ax.set_title("Switching Function over Time"); 

#Ephemeris plotting
sun_rad = 6.957e8;
eph.plot_xy(sun_rad);
eph.plot_xy_ref_orbit(sma_Earth, "Earth Orbit" );
eph.plot_xy_ref_orbit(sma_Mars, "Mars Orbit" );


print("r: ", r_0);
print("theta: ", theta_0);
print("r_dot: ", r_dot_0);
print("v_theta: ", v_theta_0 );
print("x: ", x0);
print("y: ", y0);
print("vx: ", vx0);
print("vy: ", vy0);
print("m: ", m_0);
print("Full state vector: ", arr_full_y0 );
print("Initial derivatives: ", derivs );

