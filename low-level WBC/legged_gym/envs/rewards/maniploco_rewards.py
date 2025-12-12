import torch
from isaacgym.torch_utils import *


class ManipLoco_rewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    # -------------Z1: Reward functions----------------

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.env.base_yaw_quat, self.env.ee_pos - self.env.get_ee_goal_spherical_center())
        ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.env.curr_ee_goal_sphere) * self.env.sphere_error_scale, dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - self.env.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma * 2)
        return rew, ee_pos_error

    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_cart(self):
        target_ee = self.env.get_ee_goal_spherical_center() + quat_apply(self.env.base_yaw_quat, self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error
    
    def _reward_tracking_ee_orn(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler)) * self.env.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    def _reward_arm_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, 12:-self.env.cfg.env.num_gripper_joints] * self.env.dof_vel[:, 12:-self.env.cfg.env.num_gripper_joints]), dim = 1)
        return energy, energy

    def _reward_tracking_ee_orn_ry(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler) * self.env.orn_error_scale)[:, [0, 2]]), dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    # -------------B1: Reward functions----------------

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2, action_l2

    def _reward_leg_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim = 1)
        return energy, energy

    def _reward_leg_energy_sum_abs(self):
        energy = torch.abs(torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1))
        return energy, energy
    
    def _reward_leg_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, :12] ** 2, dim=1)
        return action_l2, action_l2
    
    def _reward_leg_energy(self):
        energy = torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1)
        return energy, energy
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma), lin_vel_error

    def _reward_tracking_lin_vel_x_l1(self):
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < 1e-5
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        rew = 0*error
        rew_x = -error + torch.abs(self.env.commands[:, 0])
        rew[~zero_cmd_indices] = rew_x[~zero_cmd_indices] / (torch.abs(self.env.commands[~zero_cmd_indices, 0]) + 0.01)
        rew[zero_cmd_indices] = 0
        return rew, error

    def _reward_tracking_lin_vel_x_exp(self):
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return - error + torch.abs(self.env.commands[:, 2]), error
    
    def _reward_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_lin_vel_y_l2(self):
        squared_error = (self.env.commands[:, 1] - self.env.base_lin_vel[:, 1]) ** 2
        return squared_error, squared_error
    
    def _reward_tracking_lin_vel_z_l2(self):
        squared_error = (self.env.commands[:, 2] - self.env.base_lin_vel[:, 2]) ** 2
        return squared_error, squared_error
    
    def _reward_survive(self):
        survival_reward = torch.ones(self.env.num_envs, device=self.env.device)
        return survival_reward, survival_reward

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.env.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        return foot_contacts_z, foot_contacts_z

    def _reward_torques(self):
        # Penalize torques
        torque = torch.sum(torch.square(self.env.torques), dim=1)
        return torque, torque
    
    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim=1)
        return energy, energy

    def _reward_tracking_lin_vel_y(self):
        cmd = self.env.commands[:, 1].clone()
        lin_vel_y_error = torch.square(cmd - self.env.base_lin_vel[:, 1])
        rew = torch.exp(-lin_vel_y_error/self.env.cfg.rewards.tracking_sigma)
        return rew, lin_vel_y_error
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        return rew, rew
    
    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
        return rew, rew
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma)
        return rew, rew
    

    def _reward_ang_vel_standing(self):
        # Penalize angular velocity with respect to Z while standing
        rew = torch.square(self.env.base_ang_vel[:, 2])
        rew[self.env._get_walking_cmd_mask()] = 0.
        return rew, rew
    
    def _reward_work(self):
        work = self.env.torques * self.env.dof_vel
        abs_sum_work = torch.abs(torch.sum(work[:, :12], dim = 1))
        return abs_sum_work, abs_sum_work
    
    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[:, :12] / self.env.dt), dim=1)
        return rew, rew
    
    def _reward_dof_vel(self):
        rew = torch.sum(torch.abs(self.env.dof_vel[:, :12]), dim=1)
        return rew, rew
    
    def _reward_action_rate(self):
        action_rate = torch.sum(torch.square(self.env.last_actions - self.env.actions)[:, :12], dim=1)
        return action_rate, action_rate
    
    def _reward_action_jerk(self):
        """
        奖励平滑的动作变化，使用二阶变化（Jerk）来减少剧烈震颤。
        """
        # 计算二阶变化率 (Jerk) = 第二阶变化
        action_jerk = torch.sum(torch.square(self.env.last_actions - 2 * self.env.actions + self.env.last_last_actions)[:, :12], dim=1)
        return action_jerk, action_jerk
        
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.) # upper limit
        rew = torch.sum(out_of_limits[:, :12], dim=1)
        # print("self.env.dof_pos_limits", self.env.dof_pos_limits[:, 0])
        return rew, rew
    
    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.env.torques - self.env.last_torques)[:, :12], dim=1)
        return rew, rew
    
    def _reward_collision(self):
        rew = torch.sum(1.*(torch.norm(self.env.contact_forces[:, self.env.penalized_contact_indices, :], dim=-1) > 0.1), dim=1)
        return rew, rew
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        # rew = -dof_error
        rew[self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_walking_dof(self):
        # Penalize motion at zero commands
        # print("self.env.dof_pos", self.env.dof_pos)
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[~self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_feet_jerk(self):
        if not hasattr(self, "last_contact_forces"):            
            result = torch.zeros(self.env.num_envs).to(self.env.device)
        else:
            result = torch.sum(torch.norm(self.env.force_sensor_tensor - self.env.last_contact_forces, dim=-1), dim=-1)
        
        self.env.last_contact_forces = self.env.force_sensor_tensor.clone()
        result[self.env.episode_length_buf<50] = 0.
        return result, result
    
    def _reward_alive(self):
        return 1., 1.
    
    def _reward_feet_drag(self):
        # Penalize dragging feet
        feet_xyz_vel = torch.abs(self.env.rigid_body_state[:, self.env.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.env.foot_contacts_from_sensor * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew, rew


    def _reward_feet_contact_forces(self):
        reset_flag = (self.env.episode_length_buf > 2./self.env.dt).type(torch.float)
        forces = torch.sum((torch.norm(self.env.force_sensor_tensor, dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0), dim=-1)
        rew = reset_flag * forces
        return rew, rew
    

    def _reward_feet_contact_forces_standing(self):

        # 只在 episode 稍微稳定之后再给予奖励
        reset_flag = (self.env.episode_length_buf > 2. / self.env.dt).float()

        # 每条腿的接触力大小 [num_envs, num_feet]
        contact_forces = torch.norm(self.env.force_sensor_tensor, dim=-1)  # [N, 4]

        # 惩罚“浮空”腿：小于最小应力则惩罚
        min_required_force = self.env.cfg.rewards.min_contact_force  # e.g., 30.0
        low_force_penalty = (min_required_force - contact_forces).clamp(min=0)  # [N, 4]
        # print("contact_forces", contact_forces)

        # 只有在站立状态下 (cmd ≈ 0) 才生效
        standing_mask = (~self.env._get_walking_cmd_mask()).float().unsqueeze(-1)  # [N, 1]

        # 应用 standing mask，只对静止时生效
        rew = low_force_penalty * standing_mask * reset_flag.unsqueeze(-1)  # [N, 4]

        # 汇总成一个标量奖励 per env
        rew_total = torch.sum(rew, dim=1)  # 惩罚为负数，奖励方向向好

        return rew_total, rew_total

    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        error = torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
        return error, error
    
    def _reward_roll(self):
        # Penalize non flat base orientation
        roll = self.env._get_body_orientation()[:, 0]
        error = torch.abs(roll)
        return error, error
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1), dim=1)

        height_error = base_height - self.env.cfg.rewards.base_height_target
        rew = torch.exp(-height_error**2 / self.env.cfg.rewards.base_height_sigma)
        return rew, rew

        # return torch.abs(base_height - self.env.cfg.rewards.base_height_target), base_height
    
    
    def _reward_orientation_walking(self):
        reward, metric = self.env._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric
    
    def _reward_orientation_standing(self):
        reward, metric = self.env._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_torques_walking(self):
        reward, metric = self.env._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_torques_standing(self):
        reward, metric = self.env._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric
    
    def _reward_energy_square_walking(self):
        reward, metric = self.env._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric
    
    def _reward_energy_square_standing(self):
        reward, metric = self.env._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_base_height_walking(self):
        reward, metric = self.env._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_base_height_standing(self):
        reward, metric = self.env._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric
    
    def _reward_dof_default_pos(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error**2 / self.env.cfg.rewards.tracking_position_sigma)
        return rew, rew
    
    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        return dof_error, dof_error
    
    def _reward_tracking_lin_vel_max_x(self):
        rew = torch.where(self.env.commands[:, 0] > 0, torch.minimum(self.env.base_lin_vel[:, 0], self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5), \
                          torch.minimum(-self.env.base_lin_vel[:, 0], -self.env.commands[:, 0]) / (-self.env.commands[:, 0] + 1e-5))
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < self.env.cfg.commands.lin_vel_x_clip
        rew[zero_cmd_indices] = torch.exp(-torch.abs(self.env.base_lin_vel[:, 0]))[zero_cmd_indices]
        return rew, rew
    
    def _reward_tracking_lin_vel_max_y(self):
        rew = torch.where(self.env.commands[:, 1] > 0, torch.minimum(self.env.base_lin_vel[:, 1], self.env.commands[:, 1]) / (self.env.commands[:, 1] + 1e-5), \
                          torch.minimum(-self.env.base_lin_vel[:, 1], -self.env.commands[:, 1]) / (-self.env.commands[:, 1] + 1e-5))
        zero_cmd_indices = torch.abs(self.env.commands[:, 1]) < self.env.cfg.commands.lin_vel_y_clip
        rew[zero_cmd_indices] = torch.exp(-torch.abs(self.env.base_lin_vel[:, 1]))[zero_cmd_indices]
        return rew, rew

    def _reward_penalty_lin_vel_y(self):
        rew = -torch.square(self.env.base_lin_vel[:, 1])
        return rew, rew
    
    
    # -------------B1 Gait Control Rewards----------------
    def _reward_hip_pos(self):
        # print(self.env.hip_indices.shape)
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.hip_indices] - self.env.default_dof_pos[self.env.hip_indices]), dim=1)
        return rew, rew

    def _reward_hip_pos_standing(self):
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.hip_indices] - self.env.default_dof_pos[self.env.hip_indices]), dim=1)
        rew[self.env._get_walking_cmd_mask()] = 0.
        return rew, rew
    
    def _reward_thigh_pos(self):
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.thigh_indices] - self.env.default_dof_pos[self.env.thigh_indices]), dim=1)
        return rew, rew
    
    
    def _reward_thigh_pos_back(self):
        # 获取前两个和后两个 thigh 的索引
        front_thigh_indices = self.env.thigh_indices[:2]  # 前两个 thigh
        back_thigh_indices = self.env.thigh_indices[2:]   # 后两个 thigh

        # 获取前两个和后两个 thigh 的关节位置
        front_thigh_pos = self.env.dof_pos[:, front_thigh_indices]
        back_thigh_pos = self.env.dof_pos[:, back_thigh_indices]

        # 获取默认关节位置
        front_default_pos = self.env.default_dof_pos[front_thigh_indices]
        back_default_pos = self.env.default_dof_pos[back_thigh_indices]

        # 前两个 thigh 小于默认值时惩罚
        front_penalty = torch.where(front_thigh_pos < (front_default_pos - 0.3),
                                    torch.square(front_default_pos - front_thigh_pos),
                                    torch.zeros_like(front_thigh_pos))

        # 后两个 thigh 大于默认值时惩罚
        back_penalty = torch.where(back_thigh_pos > back_default_pos + 0.1,
                                torch.square(back_thigh_pos - back_default_pos),
                                torch.zeros_like(back_thigh_pos))

        # 计算总惩罚
        total_penalty =  torch.sum(back_penalty, dim=1)

        return total_penalty, total_penalty
    

    def _reward_calf_pos(self):
        # print("self.env.dof", self.env.dof_pos)
        # 获取前两个和后两个 calf 的索引
        front_calf_indices = self.env.calf_indices[:2]  # 前两个 calf
        back_calf_indices = self.env.calf_indices[2:]   # 后两个 calf

        # 获取前两个和后两个 calf 的关节位置
        front_calf_pos = self.env.dof_pos[:, front_calf_indices]
        back_calf_pos = self.env.dof_pos[:, back_calf_indices]

        # 获取默认关节位置
        front_default_pos = self.env.default_dof_pos[front_calf_indices]
        back_default_pos = self.env.default_dof_pos[back_calf_indices]

        # 前两个 calf 大于默认值时惩罚
        front_penalty = torch.where(front_calf_pos > front_default_pos + 0.1,
                                    torch.square(front_calf_pos - front_default_pos),
                                    torch.zeros_like(front_calf_pos))

        # # 后两个 calf 小于默认值时惩罚
        back_penalty = torch.where(back_calf_pos < back_default_pos - 0.3,
                                torch.square(back_default_pos - back_calf_pos),
                                torch.zeros_like(back_calf_pos))

        # 计算总惩罚
        total_penalty = torch.sum(front_penalty, dim=1) #+ torch.sum(back_penalty, dim=1)

        return total_penalty, total_penalty
    

    # def _reward_calf_pos_front(self):
    #     # 计算关节位置与默认位置的差值
    #     front_leg_indices = self.env.calf_indices[:2]  # 选择前腿的索引

    #     diff = self.env.dof_pos[:, front_leg_indices] - self.env.default_dof_pos[front_leg_indices]

    #     # 仅当关节位置大于默认位置时才计算惩罚
    #     diff = torch.where(diff > 0, diff, torch.zeros_like(diff))
    #     # 计算平方惩罚
    #     rew = torch.sum(torch.square(diff), dim=1)
    #     return rew, rew
    
    # def _reward_calf_pos_back(self):
    #     # 获取后腿的索引
    #     back_leg_indices = self.env.calf_indices[2:]  # 选择后腿的索引
    #     # 获取后腿的关节位置
    #     back_leg_positions = self.env.dof_pos[:, back_leg_indices]
    #     # 惩罚后腿关节位置小于 -1.6 的情况
    #     penalty = torch.where(back_leg_positions < -1.6,
    #                         torch.abs(back_leg_positions + 1.6),  # 计算偏离 -1.6 的绝对值
    #                         torch.zeros_like(back_leg_positions))  # 如果不小于 -1.6，则无惩罚
    #     # 计算总惩罚
    #     rew = torch.sum(penalty, dim=1)
    #     return rew, rew
    
    
    def _reward_tracking_contacts_shaped_force(self):
        if not self.env.cfg.env.observe_gait_commands:
            print("====observe_gait_commands is false====")
            return 0,0
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states
        # print("desired_contact", desired_contact)
        # print("foot_forces", foot_forces[0, :])
        reward = 0
        for i in range(4):
            reward +=   -(1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        
        # cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        # reward[cmd_stop_flag] = 0
        # print("reward", reward)
        return reward / 4, reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        if not self.env.cfg.env.observe_gait_commands:
            return 0,0
            
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        # print("foot_velocities", foot_velocities)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward +=   -(desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
            
        # cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        # reward[cmd_stop_flag] = 0
        return reward / 4, reward / 4
    

    def _reward_tracking_contacts_shaped_force_2(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_swing = (self.env.clock_inputs < self.env.cfg.rewards.swing_ratio).float()
        
        reward = 0
        for i in range(4):
            reward += (desired_swing[:, i]) * (
                        (foot_forces[:, i] < 1.0).float())
                        # torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4,  reward / 4
    
    
    def _reward_tracking_contacts_shaped_vel_2(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        # print("foot_forces", foot_forces[0, :])
        # foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = (self.env.clock_inputs > (1-self.env.cfg.rewards.stance_ratio)).float()
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i]) * (
                        (foot_forces[:, i] > 1.0).float())
                        # torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
                        # torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        # print("reward: ", reward, " desired_contact: ", desired_contact, " gait indices: ", self.env.gait_indices)
        # print("clock_input", self.env.clock_inputs)
        return reward / 4,  reward / 4
    
    
    # def _reward_feet_height(self):
    #     feet_height_tracking = self.env.cfg.rewards.feet_height_target

    #     if self.env.cfg.rewards.feet_height_allfeet:
    #         feet_height = self.env.rigid_body_state[:, self.env.feet_indices, 2] # All feet
    #     else:
    #         feet_height = self.env.rigid_body_state[:, self.env.feet_indices[:2], 2] # Only front feet

    #     # print("feet_height", feet_height)

    #     rew = torch.clamp(torch.norm(feet_height, dim=-1) - feet_height_tracking, max=0)

    #     # reward: encourage all feet to reach the target height
    #     # error = feet_height - feet_height_tracking
    #     # rew = -torch.mean(error ** 2, dim=-1)

    #     # print("feet_height", feet_height)
    #     cmd_stop_flag = ~self.env._get_walking_cmd_mask()
    #     rew[cmd_stop_flag] = 0

    #     return rew, rew
    
    def _reward_feet_height(self):
        feet_height_target = self.env.cfg.rewards.feet_height_target
        feet_indices = self.env.feet_indices if self.env.cfg.rewards.feet_height_allfeet else self.env.feet_indices[:2]
        
        feet_height = self.env.rigid_body_state[:, feet_indices, 2]  # Z轴高度（离地高度）
        
        # 抬腿越接近目标高度越好（主奖励）
        error = feet_height - feet_height_target
        rew = - torch.mean(error ** 2, dim=-1)

        # 加一点对称性奖励（可选）
        # symmetry_rew = - torch.std(feet_height, dim=1)
        # rew += 0.2 * symmetry_rew

        # 检查是否静止
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()

        rew[cmd_stop_flag] = 0

        return rew, rew
            
    
    def _reward_feet_height_standing(self):
        # 获取脚的高度
        feet_indices = self.env.feet_indices
        feet_height = self.env.rigid_body_state[:, feet_indices, 2]  # (num_envs, num_feet)

        # 获取静止状态的环境 mask
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()  # (num_envs,)

        # 初始化 full reward tensor
        rew = torch.zeros(self.env.num_envs, device=self.env.device)

        # 只对静止的环境计算抬腿惩罚
        if torch.any(cmd_stop_flag):
            still_feet = feet_height[cmd_stop_flag]  # (num_standing_envs, num_feet)
            penalty = torch.mean(torch.clamp(still_feet - 0.03, min=0.0), dim=1)  # 平均离地超出 2cm 的高度
            rew[cmd_stop_flag] = -penalty  # 写入对应位置

        return rew, rew
    
    def _reward_feet_height_turning(self):
        # encourage the feet height wihle turning
        feet_height_target = self.env.cfg.rewards.feet_height_target
        # feet_indices = self.env.feet_indices if self.env.cfg.rewards.feet_height_allfeet else self.env.feet_indices[:2]

        feet_indices = self.env.feet_indices[:2]
        
        feet_height = self.env.rigid_body_state[:, feet_indices, 2]  # Z轴高度（离地高度）
        
        # 抬腿越接近目标高度越好（主奖励）
        error = feet_height - feet_height_target
        rew = - torch.mean(error ** 2, dim=-1)

        # 检查是否静止
        cmd_stop_flag = ~self.env._get_turning_cmd_mask()
        rew[cmd_stop_flag] = 0

        return rew, rew


    
    
    def _reward_foot_clearance(self):
        # keep the foot height while moving
        cur_footpos_translated = self.env.foot_positions - self.env.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.env.num_envs, len(self.env.feet_indices), 3, device=self.env.device)
        cur_footvel_translated = self.env.foot_velocities - self.env.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.env.num_envs, len(self.env.feet_indices), 3, device=self.env.device)
        for i in range(len(self.env.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.env.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.env.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.env.cfg.rewards.clearance_height_target).view(self.env.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.env.num_envs, -1)
        rew = torch.sum(height_error * foot_leteral_vel, dim=1)

        cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0

        return rew, rew
    

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes

        first_contact = (self.env.feet_air_time > 0.) * self.env.contact_filt #self.env.foot_contacts_from_sensor  #self.env.contact_filt

        self.env.feet_air_time += self.env.dt

        if self.env.cfg.rewards.feet_aritime_allfeet:
            rew_airTime = torch.sum((self.env.feet_air_time - 0.4) * first_contact, dim=1)
        else:
            rew_airTime = torch.sum((self.env.feet_air_time[:, :2] - 0.4) * first_contact[:, :2], dim=1) # Only front feet
        
        rew_airTime *= self.env._get_walking_cmd_mask()  # reward for stepping for any of the 3 motions

        self.env.feet_air_time *= ~ self.env.contact_filt #self.env.foot_contacts_from_sensor  #self.env.contact_filt
        return rew_airTime, rew_airTime
    

    





