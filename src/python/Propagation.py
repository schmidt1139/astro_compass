import numpy as np;

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


