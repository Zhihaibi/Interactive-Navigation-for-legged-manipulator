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

from legged_gym.envs.base.base_config import BaseConfig

class Z1Cfg(BaseConfig):
    class env:
        num_envs = 512
        num_of_obs = 6 + 6 + 6 + 6 + 6 + 6
        num_observations = 30 + 6  #+ 10*24  # all 
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 6 # 6 theta
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds, end episode if this time is reached
        history_len = 10

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 6 # 6 theta
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            theta1 = [-2.6180,  2.6180]
            theta2 = [ 0.0000,  3.1400]  
            theta3 = [-4.7800,  0.0000]    
            theta4 = [-1.7440,  1.5700]
            theta5 = [-1.7270,  1.7270]
            theta6 = [-2.7900,  2.7900]

    class init_state:
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { 
            'joint1': 0.0,
            'joint2': 1.5,
            'joint3': -0.42,
            'joint4': -1.3,
            'joint5': 0.0,
            'joint6': 1.57,#0.0,
            'jointGripper': -0.785}

    class control:
        action_scale = [0.4, 0.6, 0.25, 0.25, 0.05, 0] # 5 degrees
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 8  # 0.005*8 = 0.04s = 25Hz for manipulator

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/z1/urdf/z1_copy.urdf'
        name = "z1"
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors

        terminate_after_distance_to_z1 = 1.0
        reach_goal_threshold = 0.05 #0.65
        # reach_goal_threshold = 0.6

        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = True # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up


    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.]

    class box:
        box_size = 0.4
        randomize_base_mass = True
        added_mass_range = [-0.5, 2.0]
        randomize_friction = True
        friction_range = [0.5, 1.2]

    class rewards:
        sigma_action_rate = 100

        class scales:
            termination = 0.0

            ## reach point
            avoid_touching_obj_nega = -10  # reach 111
            obj_approaching_direction_posi = 1.0 #1.0
            hand_position_reaching_posi = 2.0

            # ## obj approaching the target position
            obj_position_reaching_posi = 2.0
            finish_posi = 10.0

            ### normal rewards
            time_penalty_nega = -1.0
            # dof_velocities_nega = -0.5
            # delta_sum_action_nega = -0.1
            # obj_smoothness_nega = -1.0
            # dof_pos_limits_nega = -0.00
            # joint_angle_limits_nega = -0.00

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized

    class normalization:
        class obs_scales:
            dof_pos = 1.0
        clip_observations = 100.
        clip_actions = 0.6

    class noise:
        add_noise = True
        noise_level = 2.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            gravity = 0.05

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005 #200Hz 
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class Z1CfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches         mini_batch_size = self.num_envs // num_mini_batches

        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCriticRecurrent'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 100 # number of steps per environment in a episode time 
        max_iterations = 2000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'z1_pushing_small'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt