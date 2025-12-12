# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
# from time import time
import time
from warnings import WarningMessage
import numpy as np
import os
import math


from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .z1_config import Z1Cfg


class Z1Robot(BaseTask):
    def __init__(self, cfg: Z1Cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.num_obs = cfg.env.num_observations

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        self.collisions = self.cfg.asset.self_collisions

        # create box asset
        self.box_size = self.cfg.box.box_size
        asset_options = gymapi.AssetOptions()
        asset_options.density = 7  # 4.63 - 1kg  7--1.5kg  9.26--2kg
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        box_asset = self.gym.create_box(self.sim, self.box_size, self.box_size, self.box_size, asset_options)
        
        # 设置物体的摩擦力
        box_shape_props = self.gym.get_asset_rigid_shape_properties(box_asset)
        # for shape_prop in box_shape_props:
        #     shape_prop.friction = 0.5  # 设置摩擦系数 (可根据需要调整，例如 0.5, 0.8, 1.0)
        #     shape_prop.rolling_friction = 0.0  # 防止滚动摩擦力影响 (可选)
        #     shape_prop.torsion_friction = 0.0  # 扭转摩擦 (可选)
        # # 更新属性到 asset 中
        # self.gym.set_asset_rigid_shape_properties(box_asset, box_shape_props)

        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.density = 100 
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        table_asset = self.gym.create_box(self.sim, 0.9, 0.46, 0.64, asset_options) # len 0.9, width 0.46, height 0.64

        # load robot asset
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR = LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01     
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 3000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 32
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = True
        z1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # get asset properties
        self.dof_names = self.gym.get_asset_dof_names(z1_asset)
        z1_dof_props_asset = self.gym.get_asset_dof_properties(z1_asset)
        self.z1_lower_limits = z1_dof_props_asset["lower"]
        self.z1_upper_limits = z1_dof_props_asset["upper"]
        # # 计算范围的中心点
        # center = (self.z1_lower_limits + self.z1_upper_limits) / 2.0
        # # 将范围缩小 0.9 倍
        # self.z1_lower_limits = center + self.cfg.rewards.soft_dof_pos_limit * (self.z1_lower_limits - center)
        # self.z1_upper_limits = center + self.cfg.rewards.soft_dof_pos_limit * (self.z1_upper_limits - center)

        # print("=== z1_lower_limits ==== ", z1_lower_limits)
        z1_dof_props_asset["driveMode"][:].fill(gymapi.DOF_MODE_POS)  # position control for all dofs
        # kp = [64., 128., 64., 64., 64., 64., 64.]
        # kp = np.array(kp) / 25.6
        # kd = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]
        # kd = np.array(kd)/0.0128

        z1_dof_props_asset['stiffness'][:].fill(100.0)  # TODO
        z1_dof_props_asset['damping'][:].fill(2.0)      # TODO
        # Cfg.commands.p_gains_arm = [64., 128., 64., 64., 64., 64., 64.]
        # Cfg.commands.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]
        # kp: [512.0, 768.0, 768.0, 512.0, 384.0, 256.0, 512.0]  
        # kd: 25.6

        self.z1_num_dofs = self.gym.get_asset_dof_count(z1_asset) # 7

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.z1_num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.z1_num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:-1]  # remove gripper

        print("====dof_names==== ", self.dof_names)
        print("====self.default_dof_pos[i]===", self.default_dof_pos)
        print("====self.default_dof_pos_wo_gripper===", self.default_dof_pos_wo_gripper)

        self.default_dof_state = np.zeros(self.z1_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos.cpu().numpy()

        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)

        self.z1_pose = gymapi.Transform()
        self.z1_pose.p = gymapi.Vec3(0., 0., 0.64)
        self.z1_init_pose = torch.tensor([0., 0., 0.64], dtype=torch.float, device=self.device, requires_grad=False)

        self.box_pose = gymapi.Transform()
        self.box_pose.p = gymapi.Vec3(0.69, 0, self.box_size/2.0) # + 15 cm
        self.box_init_pose = torch.tensor([0.69, 0, self.box_size/2.0], dtype=torch.float, device=self.device, requires_grad=False)
        
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(-0.22, 0, 0.30)
        
        self.box_idxs = []
        self.hand_idxs = []
        self.link4_idxs = []
        self.link5_idxs = []
        self.link6_idxs = []
        self.envs = []
        self.actor_handles = []
        self.box_actor_handles = []
        self.table_actor_handles = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            # add z1
            z1_handle = self.gym.create_actor(env_handle, z1_asset, self.z1_pose, "z1", i, self.collisions, 0)
            self.actor_handles.append(z1_handle)
            # set dof properties
            self.gym.set_actor_dof_properties(env_handle, z1_handle, z1_dof_props_asset)
            # set initial dof states
            self.gym.set_actor_dof_states(env_handle, z1_handle, self.default_dof_state, gymapi.STATE_ALL)
            # set initial position targets
            self.gym.set_actor_dof_position_targets(env_handle, z1_handle, self.default_dof_pos.cpu().numpy())
            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env_handle, z1_handle, "gripperMover", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            link4_idx = self.gym.find_actor_rigid_body_index(env_handle, z1_handle, "link04", gymapi.DOMAIN_SIM)
            self.link4_idxs.append(link4_idx)
            link5_idx = self.gym.find_actor_rigid_body_index(env_handle, z1_handle, "link05", gymapi.DOMAIN_SIM)
            self.link5_idxs.append(link5_idx)
            link6_idx = self.gym.find_actor_rigid_body_index(env_handle, z1_handle, "link06", gymapi.DOMAIN_SIM)
            self.link6_idxs.append(link6_idx)

            # add box
            offset_x = torch.rand(1, device=self.device) * -0.05  # 在 [0, 0.05] 范围内随机采样
            sign = torch.randint(0, 2, (1,),  device=self.device) * 2 - 1  # 生成 [-1, 1]
            offset_y = sign * (torch.rand(1,  device=self.device) * 0.2 + 0.2)  # 在 [0.2, 0.4] / [-0.2, -0.4]
            # print("offset_x, offset_y", offset_x, offset_y)
            
            self.box_pose.p.x = self.box_pose.p.x + offset_x.item()
            self.box_pose.p.y = self.box_pose.p.y + offset_y.item()
            self.box_pose.p.z = self.box_pose.p.z + 0.01
            self.box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi/18, math.pi/18))
            # self.box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(0, 0))
            
            # set random friction
            box_rigid_shape_props = self._process_rigid_shape_props(box_shape_props, i)
            self.gym.set_asset_rigid_shape_properties(box_asset, box_rigid_shape_props)

            box_handle = self.gym.create_actor(env_handle, box_asset, self.box_pose, "box", i, self.collisions, 0)
            self.box_actor_handles.append(box_handle)
           
            # set random mass
            box_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_handle)
            box_body_props = self._box_process_rigid_body_props(box_body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, box_handle, box_body_props, recomputeInertia=True)

            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            # get global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_rigid_body_index(env_handle, box_handle, 0, gymapi.DOMAIN_SIM)
            self.box_idxs.append(box_idx)

            # add table
            # table_handle = self.gym.create_actor(env_handle, table_asset, self.table_pose, "table", i, self.collisions, 0)
            # self.table_actor_handles.append(table_handle)
            # color = gymapi.Vec3(1,1,1)
            # self.gym.set_rigid_body_color(env_handle, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

   

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.rb_states = gymtorch.wrap_tensor(self._rb_states) # all rigid body states
        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13) # 3 actors, split the box and robot
        self.robot_root_states = self._root_states[:, 0, :]  # 13: [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        self.box_root_state = self._root_states[:, 1, :] 


        print("==self.robot_root_states ==", self.robot_root_states)
        print("==self.box_root_state ==", self.box_root_state)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.z1_num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.z1_num_dofs, 2)[..., 1]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-1] # remove gripper
    
        self.box_pos = self.box_root_state[:, :3]
        self.box_rot = euler_from_quat(self.box_root_state[:, 3:7])
        self.hand_pos = self.rb_states[self.hand_idxs, :3]
        self.hand_rot = euler_from_quat(self.rb_states[self.hand_idxs, 3:7])
        self.arm_pos = self.robot_root_states[:, 0:3]

        self.box_goal_pose_world_frame = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_goal_pose_arm_frame = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)

        self.box_pose_arm_frame = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_pose_arm_frame = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_pos_hand_frame = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_goal_pos_hand_frame = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.obj_goal_vertices = torch.zeros(self.num_envs, 8, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.obj_cur_vertices = torch.zeros(self.num_envs, 8, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.action_scale = self.cfg.control.action_scale

        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_of_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.add_noise = self.cfg.noise.add_noise

        self.global_step = 0
        self.out_of_workspace_threshold = self.cfg.asset.terminate_after_distance_to_z1
        self.reach_goal_threshold = self.cfg.asset.reach_goal_threshold
        self.redraw_flag = 0
        self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device, requires_grad=False)

        self.last_joint_sum = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.arm_pos_targets = torch.zeros(self.num_envs, self.z1_num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        # self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt # = 4 * 0.005 = 0.02  1/0.02 = 50HZ
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        print("max_episode_length", self.max_episode_length)
        print("max_episode_length_s", self.max_episode_length_s)
        print("--------dt---------", self.dt)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # print("==self.actions ==", actions.shape)
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device).requires_grad_(False)
        # print("==self.actions ==", self.actions)        
        
        self.arm_pos_targets[:,:6] = self.dof_pos_wo_gripper + self.actions * self.action_scale

        self.z1_lower_limits_tensor = torch.tensor(self.z1_lower_limits, dtype=torch.float, device=self.device, requires_grad=False)
        self.z1_upper_limits_tensor = torch.tensor(self.z1_upper_limits, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_pos_targets = torch.clip(self.arm_pos_targets, self.z1_lower_limits_tensor, self.z1_upper_limits_tensor).requires_grad_(False) 
        # print("==arm_pos_targets ==", self.arm_pos_targets)
        # print("==self.z1_lower_limits_tensor ==", self.z1_lower_limits_tensor)
        # print("==self.z1_upper_limits_tensor ==", self.z1_upper_limits_tensor)

        self.render()  # step physics and render each frame
        for _ in range(self.cfg.control.decimation):
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.arm_pos_targets))
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.episode_length_buf += 1

        # prepare some quantities
        self.box_pos = self.box_root_state[:, :3]
        self.box_rot = euler_from_quat(self.box_root_state[:, 3:7])
        self.hand_pos = self.rb_states[self.hand_idxs, :3]
        self.hand_rot = euler_from_quat(self.rb_states[self.hand_idxs, 3:7])
        self.arm_pos = self.robot_root_states[:, 0:3]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # print("==env_ids ==", env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        self.global_step += 1


        if self.redraw_flag == 0:
            self.gym.clear_lines(self.viewer)
            self._draw_target_pose()
            # self._draw_goal_vertices()
            # self._draw_box_frame_axis()
            self.redraw_flag = 1


    def check_termination(self):
        """ Check if environments need to be reset
            reset if 1- time out, 2- the object is out of the worksapce 3- the object is close to the goal
            TODO: add 2
        """
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
        distance_xy = torch.norm(self.box_pos[:, :2], dim=1).requires_grad_(False) 
        out_of_workspace = distance_xy > self.out_of_workspace_threshold

        # box_pos_y = torch.abs(self.box_pos[:,1]).requires_grad_(False) 
        # reach_goal = box_pos_y > self.reach_goal_threshold 

        box_goal_pos = self.box_goal_pose_world_frame[:, 0:3]
        distance = torch.norm(self.box_pos - box_goal_pos, dim=1)
        reach_goal = distance < self.reach_goal_threshold

        self.reset_buf = self.time_out_buf  | out_of_workspace | reach_goal

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids) 
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # reset robot states
        self.redraw_flag = 0

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # self._cal_obj_goal_vertices(env_ids)
        self._reset_target_box_pose(env_ids)

        # time.sleep(0.01) # 0.1s sleep to allow the robot to settle

        # reset buffers
        self.last_actions[env_ids] = 0.

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # self.obs_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def compute_observations(self):
        """ Computes observations
        """
        self.box_pose_arm_frame = torch.cat((self.box_pos - self.arm_pos, self.box_rot), dim=-1)
        self.hand_pose_arm_frame = torch.cat((self.hand_pos - self.arm_pos, self.hand_rot), dim=-1)

        self.box_pos_hand_frame = self.box_pos - self.hand_pos
        self.box_goal_pos_hand_frame = self.box_goal_pose_world_frame[:, :3] - self.hand_pos

        self.obs_buf = torch.cat((  self.box_pose_arm_frame * 1.0,        # dim 6
                                    self.box_goal_pose_arm_frame * 1.0,   # dim 6
                                    0*self.dof_pos_wo_gripper * 1.0,   # dim 6
                                    0*self.last_actions * 1.0,         # dim 6
                                    self.hand_pose_arm_frame * 1.0,       # dim 6
                                    self.box_pos_hand_frame * 1.0,          # dim 3
                                    self.box_goal_pos_hand_frame * 1.0,         # dim 3
                                    ),dim=-1)
        
        # print("================Observation==================")
        # print("self.arm_pos", self.arm_pos)
        # print("box_pos", self.box_pos)  
        # print("hand_pos", self.hand_pos)
        # print("box_pose_arm_frame", self.box_pose_arm_frame)
        # print("box_goal_pose_arm_frame", self.box_goal_pose_arm_frame)
        # print("hand_pose_arm_frame", self.hand_pose_arm_frame)
        # print("box_pos_hand_frame", self.box_pos_hand_frame)
        # print("box_goal_pos_hand_frame", self.box_goal_pos_hand_frame)

        if self.add_noise:
            noise = torch.randn_like(self.obs_buf) * 0.02 # -0.03~0.03
            # print("==noise ==", noise)
            self.obs_buf += noise

        # add perceptive inputs if not blind
        # if self.num_privileged_obs is not None:
        #     self.privileged_obs_buf = torch.cat((self.friction_factor, self.object_mass), dim=-1)
        #     obs_buf = torch.cat((self.obs_buf, self.privileged_obs_buf), dim=-1)

        # self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        # self.obs_buf = obs_buf
        
        # create a 10 history buffer
        # self.obs_history_buf = torch.where(
        #     (self.episode_length_buf <= 1)[:, None, None], 
        #     torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
        #     torch.cat([
        #         self.obs_history_buf[:, 1:],
        #         obs_buf.unsqueeze(1)
        #     ], dim=1)
        # ) 
        # print("==self.obs_buf ==", self.obs_buf)
        # print("==self.obs_buf.shape ==", self.obs_buf.shape)


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    #------------- Callbacks --------------
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.refresh_rigid_body_state_tensor(self.sim)


    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # reset box pose
        # box_rot_rand[:, 2] = (torch.rand(self.num_envs, device=self.device)* 2 - 1) * math.pi
        with torch.no_grad():
            offset_x = torch.rand(len(env_ids), device=self.device) * 0.05 - 0.1  # 在 [0, -0.05] 范围内随机采样
            sign = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1  # 生成 [-1, 1]
            offset_y = sign * (torch.rand(len(env_ids), device=self.device) * 0.2 + 0.2)  # 在 [0.2, 0.4] / [-0.2, -0.4]

            self.box_root_state[env_ids, 0] = self.box_init_pose[0] + offset_x
            self.box_root_state[env_ids, 1] = self.box_init_pose[1] + offset_y
            self.box_root_state[env_ids, 2] = self.box_init_pose[2]
            self.box_root_state[env_ids, 3:7] = torch.tensor([0, 0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False)

            offset_z = torch.rand(len(env_ids), device=self.device) * 0.1 - 0.05  # 在 [-0.05, 0.05] 范围内随机采样
            self.robot_root_states[env_ids, 2] =  self.z1_init_pose[2] + offset_z
           
            # random_rotations = self._random_yaw_quaternion(5, env_ids)
            # self.box_root_state[env_ids, 3:7] = random_rotations
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states)) # reset root states
            self.gym.refresh_actor_root_state_tensor(self.sim)


    def _reset_target_box_pose(self, env_ids):
        box_pos_rand = torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)
        box_rot_rand = torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)
        # offset_x = torch.rand(box_pos_rand.shape[0], device=self.device) * 0.1 - 0.05  # 在 [-0.05, 0.05] 范围内随机采样
        # sign = torch.randint(0, 2, (box_pos_rand.shape[0],),  device=self.device) * 2 - 1  # 生成 [-1, 1]
        # offset_y = sign * (torch.rand(box_pos_rand.shape[0],  device=self.device) * 0.2 + 0.2)  # 在 [0.2, 0.4] / [-0.2, -0.4]
        # box_pos_rand[:, 0] = box_pos_rand[:, 0] + offset_x
        # box_pos_rand[:, 1] = box_pos_rand[:, 1] + offset_y

        box_pos_rand[:, 0] = 0.69
        box_pos_rand[:, 1] = torch.where(self.box_pos[env_ids, 1] < 0, -0.65, 0.65)
        box_pos_rand[:, 2] = 0.3
        # box_rot_rand = euler_from_quat(self.box_root_state[env_ids, 3:7])
        
        self.box_goal_pose_arm_frame[env_ids, :] = torch.cat((box_pos_rand - self.arm_pos[env_ids, :], box_rot_rand), dim=-1)
        self.box_goal_pose_world_frame[env_ids, :] = torch.cat((box_pos_rand, box_rot_rand), dim=-1)


# =================== physical setting functions =================
    def _box_process_rigid_body_props(self, props, env_id):
        if self.cfg.box.randomize_base_mass:
            rng_mass = self.cfg.box.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)
    
        return props
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.box.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.box.friction_range
                num_buckets = 1000
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        
        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones((self.num_envs, 1, 1)) 
        
        return props
    
# =================== calculate function =================    
    def _cal_obj_goal_vertices(self, env_ids):
        #  calculate the coordinates of the eight vertices of the cube based on the target pose of the box
        half_a = self.box_size / 2.0

        # 定义正方体在局部坐标系中的顶点
        local_vertices = torch.tensor([
            [-half_a, -half_a, -half_a],
            [-half_a, -half_a, half_a],
            [-half_a, half_a, -half_a],
            [-half_a, half_a, half_a],
            [half_a, -half_a, -half_a],
            [half_a, -half_a, half_a],
            [half_a, half_a, -half_a],
            [half_a, half_a, half_a]
        ], dtype=torch.float, device= self.device, requires_grad=False)

        # 提取位置和姿态
        positions = self.box_goal_pose_world_frame[env_ids, :3]
        rolls = self.box_goal_pose_world_frame[env_ids, 3]
        pitches = self.box_goal_pose_world_frame[env_ids, 4]
        yaws = self.box_goal_pose_world_frame[env_ids, 5]

        # 计算旋转矩阵
        rotation_matrices = torch.stack([euler_to_rotation_matrix(roll, pitch, yaw) for roll, pitch, yaw in zip(rolls, pitches, yaws)])

        # 计算全局坐标系中的顶点
        self.obj_goal_vertices[env_ids, :] = torch.matmul(local_vertices.unsqueeze(0), rotation_matrices.transpose(1, 2)) + positions.unsqueeze(1)

    def _cal_obj_cur_vertices(self):
        #  calculate the coordinates of the eight vertices of the cube based on the target pose of the box
        half_a = self.box_size / 2.0

        # 定义正方体在局部坐标系中的顶点
        local_vertices = torch.tensor([
            [-half_a, -half_a, -half_a],
            [-half_a, -half_a, half_a],
            [-half_a, half_a, -half_a],
            [-half_a, half_a, half_a],
            [half_a, -half_a, -half_a],
            [half_a, -half_a, half_a],
            [half_a, half_a, -half_a],
            [half_a, half_a, half_a]
        ], dtype=torch.float, device= self.device, requires_grad=False)

        # 提取位置和姿态
        positions = self.box_pose_arm_frame[:, :3]
        rolls = self.box_pose_arm_frame[:, 3]
        pitches = self.box_pose_arm_frame[:, 4]
        yaws = self.box_pose_arm_frame[:, 5]

        # 计算旋转矩阵
        rotation_matrices = torch.stack([euler_to_rotation_matrix(roll, pitch, yaw) for roll, pitch, yaw in zip(rolls, pitches, yaws)])

        # 计算全局坐标系中的顶点
        self.obj_cur_vertices = torch.matmul(local_vertices.unsqueeze(0), rotation_matrices.transpose(1, 2)) + positions.unsqueeze(1)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        self.last_reward = {}
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

            # initialize ——last_reward
            self.last_reward[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}


    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    

    def _draw_target_pose(self):
        self.center = torch.tensor([0., 0., 0.], dtype=torch.float, device=self.device, requires_grad=False)
        bbox0 = self.center + self.box_size/2
        bbox1 = self.center - self.box_size/2
        # print("==bbox0==", bbox0)
        bboxes = torch.stack([bbox0, bbox1], dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1)
        # print("==bboxes==", bboxes)

        for i in range(self.num_envs):
            bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))
            roll, pitch, yaw = self.box_goal_pose_world_frame[i, 3:6]
            quat = quat_from_euler_xyz(roll, pitch, yaw)
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            # r = gymapi.Quat(0., 0., 0., 0.)

            pose0 = gymapi.Transform(gymapi.Vec3(self.box_goal_pose_world_frame[i, 0], self.box_goal_pose_world_frame[i, 1], self.box_size/2.0), r=r)
            # pose0 = gymapi.Transform(gymapi.Vec3(0, 0, 1.), r=r)
            gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0) 
    

    def _draw_goal_vertices(self):
        for i in range(self.num_envs):
            vertices = self.obj_goal_vertices[i]
            # print("==vertices==", vertices)
            for j in range(8):
                point = vertices[j]
                # print("==point==", point)
                sphere_geom = gymutil.WireframeSphereGeometry(0.05, 2, 2, None, color=(0, 1, 0))
                pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]))
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)


    def _draw_box_frame_axis(self):
        for i in range(self.num_envs):
            axes_geom = gymutil.AxesGeometry(scale=0.5)

            roll, pitch, yaw = self.box_goal_pose_world_frame[i, 3:6]
            quat = quat_from_euler_xyz(roll, pitch, yaw)
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            # r = gymapi.Quat(0., 0., 0., 0.)
            pose0 = gymapi.Transform(gymapi.Vec3(self.box_goal_pose_world_frame[i, 0], self.box_goal_pose_world_frame[i, 1], self.box_size/2.0), r=r)
            
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose0)

            roll, pitch, yaw = self.box_pose_arm_frame[i, 3:6]
            quat = quat_from_euler_xyz(roll, pitch, yaw)
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            pose1 = gymapi.Transform(gymapi.Vec3(self.box_pose_arm_frame[i, 0], self.box_pose_arm_frame[i, 1], self.box_size/2.0), r=r)
            
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose1)


    def _get_mask_distance(self):
        # get the mask for approaching the the target position
        hand_pos = self.rb_states[self.hand_idxs, :3]
        box_pos = self.box_root_state[:,0:3]
        box_goal_pos = self.box_goal_pose_world_frame[:,0:3]
        
        d = box_goal_pos - box_pos  # 方向向量
        d_norm = d / torch.norm(d, dim=1, keepdim=True)  # 归一化
        target_position = box_pos - d_norm * (self.box_size / 2)
        distance_hand = torch.norm(hand_pos - target_position, dim=1)

        mask = (distance_hand < 0.1)
        return mask
    

    def _random_yaw_quaternion(self, degrees, env_ids):
        radians = math.radians(degrees)
        random_yaw = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * radians
        cos_yaw = torch.cos(random_yaw / 2)
        sin_yaw = torch.sin(random_yaw / 2)
        
        qx = torch.zeros_like(cos_yaw)
        qy = torch.zeros_like(cos_yaw)
        qz = sin_yaw
        qw = cos_yaw
        
        return torch.stack([qx, qy, qz, qw], dim=1)

    # def _generate_random_offset(self, num_envs, device, min_offset, max_offset, threshold):
    #     offset = torch.rand(num_envs, device=device) * (max_offset - min_offset) + min_offset
    #     while torch.any(torch.abs(offset) < threshold):
    #         offset = torch.rand(num_envs, device=device) * (max_offset - min_offset) + min_offset
    #     return offset

    #------------ reward functions----------------  

    #=================== approaching reward ===================
    def _reward_obj_approaching_direction_posi(self):
        # encourage the z1 to approach the object in the right direction
        box_goal_pos = self.box_goal_pose_world_frame[:,0:3]
        box_pos = self.box_pos.clone()
        box_pos[:, 2] = self.box_size / 2.0
        d = box_goal_pos - box_pos # 方向向量
        d_norm = d / torch.norm(d, dim=1, keepdim=True)  # 归一化
        target_position = box_pos - d_norm * (self.box_size / 2)
        target_position[: 0] = target_position[: 0]  + 0.05

        distance_hand = torch.norm(self.hand_pos - target_position, dim=1)
        reward = torch.where(distance_hand < 0.15, torch.exp(-distance_hand) + 5.0, torch.exp(-distance_hand))
        # reward = torch.exp(-distance_hand)
        # print("==distance_hand==", distance_hand)
        return reward
    
    def _reward_avoid_touching_obj_nega(self):
        hand_pos = self.hand_pos
        link6_pos = self.rb_states[self.link6_idxs, :3]
        link5_pos = self.rb_states[self.link5_idxs, :3]

        box_pos = self.box_pos
        box_size = self.box_size + 0.15

        # Calculate the boundaries of the box
        box_min = box_pos - box_size / 2.0
        box_max = box_pos + box_size / 2.0

        base_min = torch.tensor([-0.29, -0.29, 0.12], dtype=torch.float, device=self.device, requires_grad=False)
        base_max = torch.tensor([0.34, 0.34, 0.67], dtype=torch.float, device=self.device, requires_grad=False)

        # Check if hand is inside the box
        inside_box_hand = torch.all((hand_pos >= box_min) & (hand_pos <= box_max), dim=1)
        inside_box_link5 = torch.all((link5_pos >= box_min) & (link5_pos <= box_max), dim=1)
        inside_box_link6 = torch.all((link6_pos >= box_min) & (link6_pos <= box_max), dim=1)

        # base collision
        inside_base_hand = torch.all((hand_pos >= base_min) & (hand_pos <= base_max), dim=1)
        inside_base_link6 = torch.all((link6_pos >= base_min) & (link6_pos <= base_max), dim=1)

        # Combine the results
        inside_box = inside_box_hand | inside_box_link6 | inside_base_hand | inside_base_link6
        
        reach_push_z = torch.abs(hand_pos[:, 2]) > 0.6
        reach_push_x = torch.abs(hand_pos[:, 0]) < 0.45

        inside_box = inside_box & reach_push_z & reach_push_x
        reward = torch.where(inside_box, 1.0, 0.0)
        return reward
    
    def _reward_perpendicular_alignment_posi(self):
        # 获取关节6和手的位置
        link6_pos = self.rb_states[self.link6_idxs, :3]
        hand_pos = self.hand_pos

        # 获取箱子和目标点的位置
        box_pos = self.box_pos
        target_pos = self.box_goal_pose_world_frame[:, :3]

        # 计算关节6和手之间的向量
        joint6_to_hand = hand_pos - link6_pos

        # 计算箱子和目标点之间的向量
        box_to_target = target_pos - box_pos

        # 计算两个向量的点积
        dot_product = torch.sum(joint6_to_hand * box_to_target, dim=1)

        # 计算奖励，余弦值接近于零时奖励最大
        reward = torch.exp(-torch.abs(dot_product))
        return reward   


    def _reward_obj_approaching_posi(self):
        # Reward for the z1 approaching the object
        distance_hand = torch.norm(self.hand_pos - self.box_pos, dim=1)
        reward = torch.exp(-distance_hand)
        return reward

    def _reward_arm_z_position_posi(self):
        # keep z of the gripper on the middle of the object, for pushing    
        hand_pos_z = self.hand_pos[:, 2]
        box_pos_z =self.box_pos[:, 2]
        distance_hand = torch.abs(hand_pos_z - 0.3)
        reward = torch.exp(-distance_hand)
        return reward
    
    def _reward_arm_x_position_posi(self):
        # keep z of the gripper on the middle of the object, for pushing    
        hand_pos_x = self.hand_pos[:, 0]
        box_pos_x = self.box_pos[:, 0]
        distance_hand = torch.abs(hand_pos_x - 0.69)
        reward = torch.exp(-distance_hand)
        return reward
    
    
    #=================== pushing reward ===================
    def _reward_obj_pose_reaching_nega(self):
        # encourage for reaching the vertices of the object
        distance = torch.norm(self.obj_goal_vertices  - self.obj_cur_vertices, dim=2)  # 按最后一维计算距离，形状 (N, 8)
        avg_distance = torch.sum(distance, dim=1)  # 按列求平均，形状 (N,)
        # print("==avg_distance==", avg_distance)
        reward = torch.log(avg_distance + 0.1).unsqueeze(1)  # 扩展维度为 (N, 1)
        # print("==reward_obj_goal_reaching==", reward.shape)
        reward = torch.squeeze(reward)
        return reward
    
    def _reward_obj_position_reaching_posi(self):
        box_goal_pos = self.box_goal_pose_world_frame[:, 0:3]  # 目标位置
        distance = torch.norm(self.box_pos - box_goal_pos, dim=1)
        reward = torch.exp(-distance)
        return reward
    
    def _reward_hand_position_reaching_posi(self):
        box_goal_pos = self.box_goal_pose_world_frame[:, 0:3]  # 目标位置
        link6_pos = self.rb_states[self.link6_idxs, :3]  # 机械臂末端位置
        distance_hand = torch.norm(self.hand_pos - box_goal_pos, dim=1)
        distance_link6 = torch.norm(link6_pos - box_goal_pos, dim=1)
        dis_ave = (distance_hand + distance_link6) / 2.0

        reward = torch.where((self.hand_pos[:, 2] < 0.6) & (self.hand_pos[:, 0] > 0.45), torch.exp(-dis_ave), torch.zeros_like(dis_ave))   
        # reward = torch.exp(-distance)
        return reward
    
    def _reward_pushing_direction_posi(self):
        # 获取机械臂末端和 box 的位置
        box_goal_pos = self.box_goal_pose_world_frame[:, 0:3]

        # 计算 box 和目标位置之间的连线方向
        goal_direction = box_goal_pos - self.box_pos
        goal_direction_norm = goal_direction / torch.norm(goal_direction, dim=1, keepdim=True)

        # 计算机械臂末端和 box 之间的方向
        push_direction = self.box_pos - self.hand_pos
        push_direction_norm = push_direction / torch.norm(push_direction, dim=1, keepdim=True)

        # 计算方向对齐程度（点积）
        alignment = torch.sum(goal_direction_norm * push_direction_norm, dim=1)

        # 奖励与目标方向对齐的推动
        reward = torch.exp(alignment - 1.0)  # alignment 越接近 1，奖励越高
        # print("==alignment==", alignment)
        return reward
    

    def _reward_finish_posi(self):
        # Reward for finishing the task
        box_goal_pos = self.box_goal_pose_world_frame[:, 0:3]
        distance = torch.norm(self.box_pos - box_goal_pos, dim=1)
        reward = torch.where(distance < (self.reach_goal_threshold + 0.001), 10.0, 0.0)

        # box_y = torch.abs(self.box_pos[:, 1])
        # reward = torch.where(box_y > (self.reach_goal_threshold - 0.001), 10.0, 0.0)
        # print("==box_y==", box_y)
        return reward
    

    #=================== normal reward ===================
    def _reward_time_penalty_nega(self):
        # Penalize time
        return torch.ones(self.num_envs, device=self.device)
    
    def _reward_delta_sum_action_nega(self):
        # Penalize changes in actions using exp
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_dof_velocities_nega(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits_nega(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.z1_lower_limits[0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.z1_upper_limits[1]).clip(min=0.)
        reward = torch.sum(out_of_limits, dim=1)
        # print("==reward_dof_pos_limits_nega==", reward.shape)
        return reward
    
    def _reward_joint_angle_limits_nega(self):
        # Penalize joint angles too close to the limit
        joint2_out_of_limits = self.dof_pos[:, 1] - 1.5
        reward1 = torch.where(joint2_out_of_limits < 0, 1.0, 0.0)

        joint1_out_of_limits = torch.abs(self.dof_pos[:, 0]) - 1.0
        reward2 = torch.where(joint1_out_of_limits > 0, 1.0, 0.0)
        reward = reward1 + reward2
        return reward
    
    def _reward_obj_smoothness_nega(self):
        # Penalize object higher than the threshold
        return torch.where(self.box_pos[:, 2] > (self.box_size/2+0.001), 1.0, 0.0)




        
        
