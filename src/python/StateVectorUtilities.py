import numpy as np;

def non_dimensionalize( arr_y, g0, mu, T_max, ISP, TOF, l_star, m_star, t_star ):
    
    #unpack the state vector
    x, y, vx, vy, m = arr_y[:5];
    
    #initialize non-dim variables to input state vector
    x_nd, y_nd, vx_nd, vy_nd, m_nd = x, y, vx, vy, m;
    
    #initial parameters
    g0_nd = g0;
    mu_nd = mu;
    T_max_nd = T_max;
    ISP_nd = ISP;
    TOF_nd = TOF;
    
    #non-dimensionalize by length
    x_nd = x_nd / l_star;
    y_nd = y_nd / l_star;
    vx_nd = vx_nd / l_star;
    vy_nd = vy_nd / l_star;
    g0_nd = g0_nd / l_star;
    mu_nd = mu_nd / (l_star)**3;
    T_max_nd = T_max_nd / l_star;
    
    #non-dimensionalize by time
    vx_nd = vx_nd * t_star;
    vy_nd = vy_nd * t_star;
    g0_nd = g0_nd * t_star**2;
    mu_nd = mu_nd * t_star**2;
    T_max_nd = T_max_nd * t_star**2;
    ISP_nd = ISP_nd / t_star;
    TOF_nd = TOF_nd / t_star;
    
    #non-dimensionalize by mass
    m_nd = m_nd / m_star;
    T_max_nd = T_max_nd / m_star;
    
    #pack nd array
    arr_y_nd = np.array([x_nd, y_nd, vx_nd, vy_nd, m_nd]);
    
    print("l_star: ", l_star);
    print("m_star: ", m_star);
    print("t_star: ", t_star);
    print("arr_y_nd: ", arr_y_nd );
    print("g0_nd: ", g0_nd );
    print("mu_nd: ", mu_nd );
    print("T_max: ", T_max );
    print("ISP_nd: ", ISP_nd );
    print("TOF_nd: ", TOF_nd );
    print("");
    
    #return outputs
    return arr_y_nd, g0_nd, mu_nd, T_max_nd, ISP_nd, TOF_nd;
    
    