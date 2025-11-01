<div align="center">

# Interactive-Navigation-with-Learned-Arm-pushing-Controller</a>

[![Project Page](https://img.shields.io/badge/Project%20Page-6DE1D2?style=for-the-badge&logo=safari&labelColor=555555)](https://zhihaibi.github.io/interactive-push.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-F75A5A?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/pdf/2503.01474)

<p align ="center">
<img src="./Images/fig1.png" width=90%>
</p>
</div>

<font size=4>
Interactive navigation is crucial in scenarios where proactively interacting with objects can yield shorter paths, thus significantly improving traversal efficiency. Existing methods primarily focus on using the robot body to relocate large obstacles (which could be comparable to the size of a robot). However, they prove ineffective in narrow or constrained spaces where the robot's dimensions restrict its manipulation capabilities. This paper introduces a novel interactive navigation framework for legged manipulators, featuring an active arm-pushing mechanism that enables the robot to reposition movable obstacles in space-constrained environments. To this end, we develop a reinforcement learning-based arm-pushing controller with a two-stage reward strategy for large-object manipulation. Specifically, this strategy first directs the manipulator to a designated pushing zone to achieve a kinematically feasible contact configuration. Then, the end effector is guided to maintain its position at appropriate contact points for stable object displacement while preventing toppling. The simulations validate the robustness of the arm-pushing controller, showing that the two-stage reward strategy improves policy convergence and long-term performance. Real-world experiments further demonstrate the effectiveness of the proposed navigation framework, which achieves shorter paths and reduced traversal time. 
</font>

## Results

<p align ="center">
<img src="./Images/Interactive_Nav_IROS_2025_final.gif" width=100%>
</p>

<font size=4>
We deploy our method on a legged manipulator in a 6.8 m × 8 m cluttered indoor environment.
</font>

## Quick Start For the Simple Case
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2.  Clone this repository
3. Install pytorch 1.10 with cuda-11.3: (change according to your cuda version)
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
4. Install Isaac Gym
   - `cd src/isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
5. Install rsl_rl (PPO implementation)
   -  `cd src/rsl_rl && git checkout v1.0.2 && pip install -e .` 
6. Install legged_gym
  
   - `cd src && pip install -e .`
7. Training:
   
     ```python src/legged_gym/scripts/train.py --task=z1```



## Implementation
- [√] Training code
- [ ] Tutorial
- [ ] Planner code: Modified hybrid A* 


## Authors

- [@Zhihai Bi](zbi217@connect.hkust-gz.edu.cn)

Feel free to contact me if you have any questions regarding the implementation of the algorithm.
