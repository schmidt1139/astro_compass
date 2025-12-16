class SACRolloutData:
    def __init__(self):
        self.arr_time = []
        self.arr_reward_tot = []
        self.arr_reward = []
        self.arr_throttle = []
        self.arr_alpha_x = []
        self.arr_alpha_y = []
        self.arr_x = []
        self.arr_y = []
        self.arr_vx = []
        self.arr_vy = []
        self.arr_sma = []
        self.arr_sma_target = []
        self.arr_ecc = []
        self.arr_ecc_target = []
        self.arr_ecc_max = []
        self.arr_reward_mass = []
        self.arr_reward_distance = []
        self.sum_reward = 0.0

    def add_step(
        self,
        time,
        reward,
        throttle,
        alpha_x,
        alpha_y,
        x,
        y,
        vx,
        vy,
        sma,
        sma_target,
        ecc,
        ecc_target,
        ecc_max,
        reward_mass,
        reward_distance,
    ):
        self.arr_time.append(time)  # convert to days
        self.arr_reward.append(reward)
        self.arr_throttle.append(throttle)
        self.arr_alpha_x.append(alpha_x)
        self.arr_alpha_y.append(alpha_y)
        self.arr_x.append(x)
        self.arr_y.append(y)
        self.arr_vx.append(vx)
        self.arr_vy.append(vy)
        self.arr_sma.append(sma)
        self.arr_sma_target.append(sma_target)
        self.arr_ecc.append(ecc)
        self.arr_ecc_target.append(ecc_target)
        self.arr_ecc_max.append(ecc_max)
        self.sum_reward += reward
        self.arr_reward_tot.append(self.sum_reward)
        self.arr_reward_mass.append(reward_mass)
        self.arr_reward_distance.append(reward_distance)


class SACRolloutData_TBR:
    def __init__(self):
        self.arr_time = []
        self.arr_reward_tot = []
        self.arr_reward = []
        self.arr_throttle = []
        self.arr_alpha_x = []
        self.arr_alpha_y = []
        self.arr_x = []
        self.arr_y = []
        self.arr_vx = []
        self.arr_vy = []
        self.arr_m = []
        self.arr_x_target = []
        self.arr_y_target = []
        self.arr_vx_target = []
        self.arr_vy_target = []
        self.arr_ttg = []
        self.arr_pos_reward = []
        self.arr_vel_reward = []
        self.arr_mass_reward = []
        self.sum_reward = 0.0

    def add_step(
        self,
        time,
        reward,
        throttle,
        alpha_x,
        alpha_y,
        x,
        y,
        vx,
        vy,
        m,
        x_target,
        y_target,
        vx_target,
        vy_target,
        ttg,
        pos_reward,
        vel_reward,
        mass_reward,
    ):
        self.arr_time.append(time)  # convert to days
        self.arr_reward.append(reward)
        self.arr_throttle.append(throttle)
        self.arr_alpha_x.append(alpha_x)
        self.arr_alpha_y.append(alpha_y)
        self.arr_x.append(x)
        self.arr_y.append(y)
        self.arr_vx.append(vx)
        self.arr_vy.append(vy)
        self.arr_m.append(m)
        self.arr_x_target.append(x_target)
        self.arr_y_target.append(y_target)
        self.arr_vx_target.append(vx_target)
        self.arr_vy_target.append(vy_target)
        self.arr_ttg.append(ttg)
        self.arr_pos_reward.append(pos_reward)
        self.arr_vel_reward.append(vel_reward)
        self.arr_mass_reward.append(mass_reward)
        self.sum_reward += reward
        self.arr_reward_tot.append(self.sum_reward)


class SACRolloutData_TBR_polar:
    def __init__(self):
        self.arr_time = []
        self.arr_reward_tot = []
        self.arr_reward = []

        self.arr_throttle = []
        self.arr_alpha_r = []
        self.arr_alpha_theta = []

        self.arr_rad = []
        self.arr_cos_theta = []
        self.arr_sin_theta = []
        self.arr_v = []
        self.arr_v_r_unit = []
        self.arr_v_t_unit = []
        self.arr_mass = []

        self.arr_rad_f = []
        self.arr_cos_theta_f = []
        self.arr_sin_theta_f = []
        self.arr_v_f = []
        self.arr_v_r_f_unit = []
        self.arr_v_t_f_unit = []
        self.arr_ttg = []

        self.arr_pos_reward = []
        self.arr_vel_reward = []
        self.arr_mass_reward = []
        self.arr_throttle_reward = []
        self.sum_reward = 0.0

        self.arr_position_res = []
        self.arr_velocity_res = []

    def add_step(
        self,
        time,
        reward,
        throttle,
        alpha_r,
        alpha_theta,
        rad,
        cos_theta,
        sin_theta,
        v,
        v_r_unit,
        v_t_unit,
        m,
        rad_f,
        cos_theta_f,
        sin_theta_f,
        v_f,
        v_r_f_unit,
        v_t_f_unit,
        ttg,
        pos_reward,
        vel_reward,
        mass_reward,
        throttle_reward,
        position_res,
        velocity_res,
    ):
        self.arr_time.append(time)  # convert to days
        self.arr_reward.append(reward)
        self.arr_throttle.append(throttle)
        self.arr_alpha_r.append(alpha_r)
        self.arr_alpha_theta.append(alpha_theta)
        self.arr_rad.append(rad)
        self.arr_cos_theta.append(cos_theta)
        self.arr_sin_theta.append(sin_theta)
        self.arr_v.append(v)
        self.arr_v_r_unit.append(v_r_unit)
        self.arr_v_t_unit.append(v_t_unit)
        self.arr_mass.append(m)
        self.arr_rad_f.append(rad_f)
        self.arr_cos_theta_f.append(cos_theta_f)
        self.arr_sin_theta_f.append(sin_theta_f)
        self.arr_v_f.append(v_f)
        self.arr_v_r_f_unit.append(v_r_f_unit)
        self.arr_v_t_f_unit.append(v_t_f_unit)
        self.arr_ttg.append(ttg)
        self.arr_pos_reward.append(pos_reward)
        self.arr_vel_reward.append(vel_reward)
        self.arr_mass_reward.append(mass_reward)
        self.arr_throttle_reward.append(throttle_reward)

        self.sum_reward += reward
        self.arr_reward_tot.append(self.sum_reward)
        self.arr_position_res.append(position_res)
        self.arr_velocity_res.append(velocity_res)
