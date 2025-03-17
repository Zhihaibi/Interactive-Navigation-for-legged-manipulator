# Interactive-Navigation-with-Learned-Arm-pushing-Controller


<p align ="center">
<img src="./Images/fig1.png" width=90%>
</p>

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


## Implementation
- [✔] Training code
- [ ] Planner code 
- [ ] Tutorial

## Authors

- [@Zhihai Bi](zbi217@connect.hkust-gz.edu.cn)

Feel free to contact me if you have any questions regarding the implementation of the algorithm.
