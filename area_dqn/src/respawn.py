#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import time
import random
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.subgoal_position = Pose()
        self.goal_position.position.x = 0
        self.goal_position.position.y = 0
        self.subgoal_position.position.x = 0
        self.subgoal_position.position.x = 0
        self.respawn()

    def respawn(self):
        #remember = 0
        while not rospy.is_shutdown():
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):

                if model.name[i] == 'goal':
                    obstacle = ModelState()
                    obstacle.model_name = model.name[i]
                    obstacle.pose = model.pose[i]
                    obstacle.pose.position.x = self.goal_position.position.x
                    obstacle.pose.position.y = self.goal_position.position.y
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)

                if model.name[i] == 'subgoal':
                    obstacle = ModelState()
                    obstacle.model_name = model.name[i]
                    obstacle.pose = model.pose[i]
                    obstacle.pose.position.x = self.subgoal_position.position.x
                    obstacle.pose.position.y = self.subgoal_position.position.y
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)

    def getSubgoalPosition(self, x, y):
        self.subgoal_position.position.x = x
        self.subgoal_position.position.y = y

    def getPosition(self):
        goal_x = -1.5
        goal_y = -4.5 + random.randrange(0,10) / 10.0
        position_check = False
                
        self.goal_position.position.x = goal_x
        self.goal_position.position.y = goal_y
        return goal_x, goal_y

def main():
    rospy.init_node('combination_obstacle_4')
    
    try:
        respawn = Respawn()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
