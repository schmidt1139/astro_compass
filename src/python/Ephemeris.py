import numpy as np;
import matplotlib.pyplot as plot;


class Ephemeris():
    
    def __init__(self):
        
        #initialize an empty ephemeris object
        
        self.arr_et = np.array([]);
        self.arr_x = np.array([]);
        self.arr_y = np.array([]);
        self.arr_vx = np.array([]);
        self.arr_vy = np.array([]);
        self.arr_m = np.array([]);
        self.num_vectors = 0;
        
    def add_data(self,et,x,y,vx,vy,m):
        
        self.arr_et = np.append( self.arr_et, et );
        self.arr_x = np.append( self.arr_x, x );
        self.arr_y = np.append( self.arr_y, y );
        self.arr_vx = np.append( self.arr_vx, vx );
        self.arr_vy = np.append( self.arr_vy, vy );
        self.arr_m = np.append( self.arr_m, m );
        self.num_vectors = self.num_vectors + 1;
        
    def add_polar_data(self,et,r,theta,r_dot,v_theta,m):
        
        #convert polar coordinates to cartesian
        x = r * np.cos(theta);
        y = r * np.sin(theta);
        vx = r_dot * np.cos(theta) - v_theta * r * np.sin(theta);
        vy = r_dot * np.sin(theta) + v_theta * r * np.cos(theta);
        
        self.arr_et = np.append( self.arr_et, et );
        self.arr_x = np.append( self.arr_x, x );
        self.arr_y = np.append( self.arr_y, y );
        self.arr_vx = np.append( self.arr_vx, vx );
        self.arr_vy = np.append( self.arr_vy, vy );
        self.arr_m = np.append( self.arr_m, m );
        self.num_vectors = self.num_vectors + 1;
        
    def plot_xy(self, radius_central_body):
        
        arr_x_cb = np.array([]);
        arr_y_cb = np.array([]);
        
        max_x = max(abs(self.arr_x));
        max_y = max(abs(self.arr_y));
        
        max_lim = 1.1* max( [max_x,max_y] );
        
        pts = 1000;
        
        #plot central body
        for i in range(0,pts):
            
            theta = 2*np.pi*i / pts;
            x_cb = radius_central_body * np.cos(theta);
            y_cb = radius_central_body * np.sin(theta);
            
            arr_x_cb = np.append( arr_x_cb, x_cb);
            arr_y_cb = np.append( arr_y_cb, y_cb);
        
        fig, ax = plot.subplots(figsize=(6, 6));
        
        ax.set_aspect("equal");
        
        ax.plot( self.arr_x, self.arr_y, label="Trajectory" );
        ax.plot( arr_x_cb, arr_y_cb, label="Central Body" );
        
        ax.set_title("Trajectory");
        ax.set_xlabel("X [km]");
        ax.set_ylabel("Y [km]");
        ax.set_xlim([-max_lim,max_lim]);
        ax.set_ylim([-max_lim,max_lim]);
        ax.legend();
        ax.grid(False);
        
        self.fig_xy = fig;
        self.ax_xy = ax;
        
        return fig;
    
    def plot_xy_ref_orbit(self, orbit_sma, label ):
        
        fig = self.fig_xy;
        ax = self.ax_xy;
        
        arr_x_ref = np.array([]);
        arr_y_ref = np.array([]);
        
        pts = 1000;
        
        #plot central body
        for i in range(0,pts):
            
            theta = 2*np.pi*i / pts;
            x_ref = orbit_sma * np.cos(theta);
            y_ref = orbit_sma * np.sin(theta);
            
            arr_x_ref = np.append( arr_x_ref, x_ref);
            arr_y_ref = np.append( arr_y_ref, y_ref);
            
        
        ax.plot( arr_x_ref, arr_y_ref, label=label, linestyle='dashed' );
        ax.legend(loc='upper left');
        
        self.fig_xy = fig;
        self.ax_xy = ax;
        
        return self.fig_xy;
        
        
        #end def __init__(self):