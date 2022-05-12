# HRL_AGV_Navigation_MTON
A navigation algorithm for Deep Reinforcement Learning developed using the Hierarchical framework.

# MTON Framework
<img src="https://user-images.githubusercontent.com/99716048/167999878-f1f41c1e-9cce-47dc-9e83-ff74dc5c0e2b.png" width=750 alt="MTON"/>

## OSM
The OSM take the statistics of a short immediate history to figure the most  probable path. The dynamic environment navigation model of our HRL framework superimposes the grid map in the environment in the past period of time according to the weight difference before and after the time axis and the fused grayscale image is used as the state input of the upper model. The output is taken to weight the grid for the ASS.<br>
<img src="https://user-images.githubusercontent.com/99716048/168080440-4b799e07-599f-42cb-96fa-d0a5cdb6da51.png" width=500 alt="OSM"/>

## ASS
ASS is used to select the best sub-goal. It takes the weight from the OSM to overcome the local-maximum problem. The ASS reinforcement learning model takes the current grid map and also the fusion map as the state input and use a modified A* algorithm as a reward in the training, finally generate visionary navigation sub-goals. <br>
<img src="https://user-images.githubusercontent.com/99716048/168080473-c83a921b-ff78-4067-bef0-0d1c7427a373.png" width=500 alt="ASS"/>

## LPP
LPP is trained to navigate from the current location to the sub-goal from the ASS. The LPP model is responsible for the speed and foot speed control of the AGV and here we use Soft-Actor-Critic (SAC) to train the model for local path planner.<br>
<img src="https://user-images.githubusercontent.com/99716048/168080523-57cb01f9-6bb6-4680-bf4a-a6ebeb268d18.png" width=500 alt="LPP"/>

# MTON Performance
https://user-images.githubusercontent.com/99716048/168000222-64b6269c-1bd6-4b92-9ef6-36b177af919d.mp4
