import numpy as np;

def spacecraft_EOM_radial_2D_EB( self,t,y,params ):
    
    '''
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
    '''
    
    
    #get the set of parameters
    params = params.astype(np.float32);
    num_params = len(params);
    
    #check parameter length
    if( num_params != 6 ):
        raise Exception('Invalid number of parameters');
    
    #unpack the parameters
    mu_cb = params[0];      #Central body gravitational parameter (km^3/s^2)
    radius_cb = params[1];  #Central body radius (km)
    C1 = params[2];         #Maximum spacecraft thrust
    C2 = params[3];         #Specific impulse
    u = params[4];          #Spacecraft throttle input (action/control)
    beta = params[5];       #Spacecraft thrust angle
    g0 = 9.80665;           #Gravity at sea level (m/s^2)
    
    #unpack state vector
    r       = y[0];         #radius
    theta   = y[1];         #theta
    r_dot   = y[2];         #r_dot
    v_theta = y[3];         #v_theta
    m       = y[4];         #mass
    
    #calc derivatives
    d_r         = r_dot;
    d_theta     = v_theta / r;
    d_r_dot     = v_theta**2 / r - mu_cb / r**2 + C1 / m * u * np.sin(beta);
    d_v_theta   = - r_dot * v_theta / r + C1 / m * u * np.cos(beta);
    d_m         = - C1 * u / g0 / C2; #kg/s
    
    #construct derivative array
    dy = np.zeros(5, dtype = np.float32 );
    
    #assign derivatives if the spacecraft has not impacted central body, 
    #otherwise keep derivatives at zero
    if ( r > radius_cb ):
        dy[0] = d_r;
        dy[1] = d_theta;
        dy[2] = d_r_dot;
        dy[3] = d_v_theta;
        dy[4] = d_m;
    
        
    #return derivative array
    return dy.astype(np.float32);

#method for calculating spacecraft EOM for Hohmann transfer env   
def spacecraft_EOM_f_2D_2B( self,t,y,params ):
    
    '''
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
    '''
    
    dy = np.zeros(4, dtype = np.float32 );
    params = params.astype(np.float32);
    
    num_params = len(params);
    
    if ( num_params == 4 ):
        mu_cb = params[0];
        radius_cb = params[1];
        x_cb = params[2];
        y_cb = params[3];
    else:
        raise Exception('Invalid number of parameters');
        
    #unpack the state vector
    x_sc = y[0];
    y_sc = y[1];
    vx_sc = y[2];
    vy_sc = y[3];
    
    #determine relative position
    x_rel = x_sc - x_cb;
    y_rel = y_sc - y_cb;
    
    #relative position magnitude
    r_rel = ( x_rel**2 + y_rel**2 )**(0.5);
    
    #define a flag to track if the spacecraft has collided with the central body
    flag_collision = False;
    
    if ( r_rel < radius_cb ):
        
        flag_collision = True;
        
        dy[0] = 0.0;
        dy[1] = 0.0;
        dy[2] = 0.0;
        dy[3] = 0.0;
        
        return dy.astype(np.float32);
    
    else:
        
        x_hat_rel = x_rel / r_rel;
        y_hat_rel = y_rel / r_rel;
        
        ddx = - mu_cb * x_hat_rel / r_rel**2;
        ddy = - mu_cb * y_hat_rel / r_rel**2;
        
        dy[0] = vx_sc;
        dy[1] = vy_sc;
        dy[2] = ddx;
        dy[3] = ddy;
                
        return dy.astype(np.float32);


