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
        self.num_vectors = 0;
        
    def add_data(self,et,x,y,vx,vy):
        
        self.arr_et = np.append( self.arr_et, et );
        self.arr_x = np.append( self.arr_x, x );
        self.arr_y = np.append( self.arr_y, y );
        self.arr_vx = np.append( self.arr_vx, vx );
        self.arr_vy = np.append( self.arr_vy, vy );
        self.num_vectors = self.num_vectors + 1;
        
    def plot_xy(self, radius_central_body):
        
        arr_x_cb = np.array([]);
        arr_y_cb = np.array([]);
        
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
        ax.legend();
        ax.grid(False);
        
        return fig;
        
        
        #end def __init__(self):