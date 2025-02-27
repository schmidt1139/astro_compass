import numpy as np;

class Spacecraft:
    
    def __init__(self, 
                 x=0.0, y=227939366, vx=23.8815223674669, vy=0.0,
                 mass=1000.0
                 ):
        
        #State vector coordinates
        self.x = x;
        self.y = y;
        self.vx = vx;
        self.vy = vy;
        self.mass = mass;
        
        
        
    def calc_Planar_OE(self,x_cb,y_cb,vx_cb,vy_cb,mu_cb):
        
        #determine coordinates relative to central body
        x_rel = self.x - x_cb;
        y_rel = self.y - y_cb;
        vx_rel = self.vx - vx_cb;
        vy_rel = self.vy - vy_cb;
        
        #position and velocity magnitudes
        r = ( x_rel**2 + y_rel**2 )**0.5;
        v = ( vx_rel**2 + vy_rel**2 )**0.5;
        
        #spacecraft position, vel, and z vectors
        sc_pos = np.array([ x_rel, y_rel, 0.0 ]);
        sc_vel = np.array([ vx_rel, vy_rel, 0.0 ]);
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
        
        #If e is zero, we will get an error dividing by zero, so the ecc vector
        #is set at {0,0,0} if the magnitude is zero.
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
