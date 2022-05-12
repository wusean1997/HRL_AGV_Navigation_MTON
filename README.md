# HRL_AGV_Navigation_MTON
A navigation algorithm for Deep Reinforcement Learning developed using the Hierarchical framework.

# MTON Framework
![gitp](https://user-images.githubusercontent.com/99716048/167999878-f1f41c1e-9cce-47dc-9e83-ff74dc5c0e2b.png)  

## OSM
The OSM take the statistics of a short immediate history to figure the most  probable path. The dynamic environment navigation model of our HRL framework superimposes the grid map in the environment in the past period of time according to the weight difference before and after the time axis and the fused grayscale image is used as the state input of the upper model. The output is taken to weight the grid for the ASS.
![OSM](https://user-images.githubusercontent.com/99716048/168078157-a891fd47-8c67-45c0-b548-39bfd78f8f04.png)

## ASS
The OSM take the statistics of a short immediate history to figure the most probable path. The dynamic environment navigation model of our HRL framework superimposes the grid map in the environment in the past period of time according to the weight difference before and after the time axis and the fused grayscale image is used as the state input of the upper model. The output is taken to weight the grid for the ASS
![ASS](https://user-images.githubusercontent.com/99716048/168078338-7c76aca4-29e1-43bf-9a13-f7d50a82ae48.png)

## LPP
LPP is trained to navigate from the current location to the sub-goal from the ASS. The LPP model is responsible for the speed and foot speed control of the AGV and here we use Soft-Actor-Critic (SAC) to train the model for local path planner. 
![LPP](https://user-images.githubusercontent.com/99716048/168078398-17dcc043-157e-4b15-b337-809b76512c4b.png)

# MTON Performance
https://user-images.githubusercontent.com/99716048/168000222-64b6269c-1bd6-4b92-9ef6-36b177af919d.mp4
