import numpy as np;

class Spacecraft:
    
    def __init__(self, 
                 x=0.0, y=227939366, vx=23.8815223674669, vy=0.0,
                 mass=1000.0
                 ):
        
        #State vector coordinates
        self.x = x;
        self.y = y;
        self.vx = self.vx;
        self.vy = self.vy;
        self.mass = mass;
        
            
    #static method for calculating spacecraft EOM for Hohmann transfer env   
    def spacecraft_EOM_f_2D_2B( t,y,params ):
        
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
            

        
        
    def calc_Planar_OE(x,y,vx,vy,mu_cb):
        
        #position and velocity magnitudes
        r = ( x**2 + y**2 )**0.5;
        v = ( vx**2 + vy**2 )**0.5;
        
        #spacecraft position, vel, and z vectors
        sc_pos = np.array([ x, y, 0.0 ]);
        sc_vel = np.array([ vx, vy, 0.0 ]);
        z_hat = np.array([ 1.0, 0.0, 0.0 ]);
        r_hat = sc_pos / r;
        
        #angular momentum
        h_vec = np.cross( sc_pos, sc_vel );
        h = np.linalg.norm(h_vec);
        h_hat = h_vec / h;
        
        #node line
        N = np.cross( z_hat, h_hat );
        N_hat = N / np.linalg.norm(N);
        
        #specific energy
        eps = v**2 / 2 + mu_cb/r;
        
        #eccentricity vector
        e_vec = np.cross(sc_vel,h_vec) / mu_cb - sc_pos/r;
        e = np.linalg.norm(e_vec);
        
        if ( e == 0.0 ):
            e_hat = e_vec*0.0;
        else:
            e_hat = e_vec/e;
        
        
        #semi major axis
        rp = h**2 / mu_cb / ( 1 + e * np.cos(0) );
        ra = h**2 / mu_cb / ( 1 + e * np.cos( np.pi ) );
        a = 1/2 * ( rp + ra );
        
        #argument of periapsis
        if ( e_vec[2] >= 0.0 ):
            w = np.acos( np.dot( N_hat, e_hat ) );
        else:
            w = 2 * np.pi - np.acos( np.dot( N_hat, e_hat ) );
        
        w_deg = np.rad2deg(w);
        
        #true anomaly - extra error handling included,
        #mainly needed for hyperbolic instances
        if ( np.dot( sc_pos, sc_vel ) >= 0.0 ):
            
            dotp = np.dot(e_hat,r_hat);
            if (dotp < -1 ):
                dotp = -1;
            elif (dotp > 1):
                dotp = 1;    
                
            theta = np.acos( dotp );
                
        else:
            
            #check acos domain
            dotp = np.dot(e_hat,r_hat);
            if (dotp < -1 ):
                dotp = -1;
            elif (dotp > 1):
                dotp = 1;
                
            theta = 2 * np.pi - np.acos( dotp );
            

        
        theta_deg = np.rad2deg(theta);
        
        # print(a)
        # print(e)
        # print(w_deg)
        # print(theta_deg)
        
        return a, e, w, theta;
