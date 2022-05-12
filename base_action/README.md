# Sean_sac

# Reinforcement Learning with Soft-Actor Critic

For this project, my goal is to create a Deep reinforcement learning Code that can avoid obstacles while trying to get to a target

## Base Idea (DQN)
- https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning

## Libraries
### [Cuda 10.1]
https://developer.nvidia.com/accelerated-computing-toolkit

### [Pytorch 1.0.1]
https://pytorch.org/

#### Install
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

### [ROS]
https://morrison.gitbook.io/ros/ros/ros-install

### [Turtlebot] 
https://morrison.gitbook.io/ros/ros/turtlebot3

You can find the packages the I used here:
- https://github.com/ROBOTIS-GIT/turtlebot3
- https://github.com/ROBOTIS-GIT/turtlebot3_msgs
- https://github.com/ROBOTIS-GIT/turtlebot3_simulations

```
cd ~/catkin_ws/src/
git clone {link_git}
cd ~/catkin_ws && catkin_make
```

To install my package you will do the same from above.

### [Gazebo(8)] & [ROS(kinetic-desktop)]
https://morrison.gitbook.io/ros/ros/gazebo

## Visualize

### [Comet.ml]
https://www.comet.ml/site/

### [Visdom]
```
pip install visdom
python -m visdom.server
http://localhost:8097
```

### [TensorBoardX]
```
pip install tensorboard
pip install tensorboardx
tensorboard --logdir runs
```
​http://localhost:6006/​


## Run Code
First to run:
```
sh gazebo.sh
```
In another terminal run:
```
sh train.sh
```


## Github

### SAC
1. https://github.com/pranz24/pytorch-soft-actor-critic
2. https://github.com/ku2482/soft-actor-critic.pytorch
3. https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/5_SAC
4. https://spinningup.openai.com/en/latest/algorithms/sac.html
