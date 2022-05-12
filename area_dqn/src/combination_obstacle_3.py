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

class Combination():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def setPoint(self, x, y, m):
        nor = 0.05
        que = 0.02
        quick = 1.0
        #A
        if (x <= 8.0 and x > 0) and (y < -3.0 and y >= -5.0):
            x = x - nor
            y = y
        #A'
        if (x <= 0.0 and x > -8.0) and (y < -3.0 and y >= -5.0):
            x = x - nor
            y = y
        #B'
        if (x <= -6.0 and x > -8.0) and (y < -1.0 and y >= -3.0):
            x = x
            y = y + nor
        #B
        if (x <= -8.0 and x >= -9.0) and (y < -1.0 and y >= -5.0):
            x = x 
            y = y + nor
        #C
        if (x < -6.0 and x >= -9.0) and (y <= 0.0 and y >= -1.0):
            x = x + nor
            y = y
        #D
        if (x < -2.5 and x >= -6.0) and (y <= 0.0 and y >= -2.0):
            x = x + nor
            y = y
        #E
        if (x < 0.0 and x >= -2.5) and (y < 0.0 and y >= -3.0):
            x = x
            y = y + nor
        #F
        if (x < 1.0 and x >= -4.0) and (y < 4.0 and y >= 0.0):
            x = x
            y = y + nor
        #G
        if (x <= 0.0 and x >= -4.0) and (y <= 5.0 and y >= 4.0):
            x = x + nor
            y = y
        #H
        if (x <= 6.5 and x >= 0.0) and (y <= 5.0 and y > 0.0):
            if m == "A"  and (x <= 2.0 and x >= 1.5):
                if y > 2.0:
                    x = x
                    y = y - nor
                else:
                    x = x
                    y = y - que
            elif m == "B" and (x <= 4.0 and x >= 3.5):
                if y > 2.0:
                    x = x
                    y = y - nor
                else:
                    x = x
                    y = y - que
            elif m == "C" and x >= 5.5:
                if y > 2.0:
                    x = x
                    y = y - nor
                else:
                    x = x
                    y = y - que
            else:
                x = x + nor
                y = y
        #I
        if (x <= 6.0 and x >= 0.0) and (y <= 0.5 and y > -2.5):
            x = x
            y = y - nor
        #J
        if (x < 8.0 and x >= 0.0) and (y <= -2.5 and y > -3.0):
            x = x + nor
            y = y
        #K
        if x >= 8.0 and (y <= 0.0 and y > -4.5):
            x = x
            y = y - nor
        #L
        if x > 8.0 and y <= -4.5:
            x = x - nor
            y = y
        return x,y           

    def moving(self):
        while not rospy.is_shutdown():
            obstacle = ModelState()
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                m = "C"
                random_item = [-0.001, -0.0005, 0, 0.0005, 0.001]
                

                if model.name[i] == 'obstacle_11':
                    obstacle.model_name = 'obstacle_11'
                    obstacle.pose = model.pose[i]
                    index_x = random.randrange(0, 5)
                    index_y = random.randrange(0, 5)
                    obstacle.pose.position.x, obstacle.pose.position.y = self.setPoint(obstacle.pose.position.x, obstacle.pose.position.y, m)
                    obstacle.pose.position.x = obstacle.pose.position.x + random_item[index_x]
                    obstacle.pose.position.y = obstacle.pose.position.y + random_item[index_y]
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)
                
                if model.name[i] == 'obstacle_12':
                    obstacle.model_name = 'obstacle_12'
                    obstacle.pose = model.pose[i]
                    random_item = [-0.005, -0.0025, 0, 0.005, 0.0025]
                    index_x = random.randrange(0, 5)
                    index_y = random.randrange(0, 5)
                    obstacle.pose.position.x, obstacle.pose.position.y = self.setPoint(obstacle.pose.position.x, obstacle.pose.position.y, m)
                    obstacle.pose.position.x = obstacle.pose.position.x + random_item[index_x]
                    obstacle.pose.position.y = obstacle.pose.position.y + random_item[index_y]
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)

                if model.name[i] == 'obstacle_13':
                    obstacle.model_name = 'obstacle_13'
                    obstacle.pose = model.pose[i]
                    index_x = random.randrange(0, 5)
                    index_y = random.randrange(0, 5)
                    obstacle.pose.position.x, obstacle.pose.position.y = self.setPoint(obstacle.pose.position.x, obstacle.pose.position.y, m)
                    obstacle.pose.position.x = obstacle.pose.position.x + random_item[index_x]
                    obstacle.pose.position.y = obstacle.pose.position.y + random_item[index_y]
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)
                
                if model.name[i] == 'obstacle_14':
                    obstacle.model_name = 'obstacle_14'
                    obstacle.pose = model.pose[i]
                    index_x = random.randrange(0, 5)
                    index_y = random.randrange(0, 5)
                    obstacle.pose.position.x, obstacle.pose.position.y = self.setPoint(obstacle.pose.position.x, obstacle.pose.position.y, m)
                    obstacle.pose.position.x = obstacle.pose.position.x + random_item[index_x]
                    obstacle.pose.position.y = obstacle.pose.position.y + random_item[index_y]
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)  

                if model.name[i] == 'obstacle_15':
                    obstacle.model_name = 'obstacle_15'
                    obstacle.pose = model.pose[i]
                    index_x = random.randrange(0, 5)
                    index_y = random.randrange(0, 5)
                    obstacle.pose.position.x, obstacle.pose.position.y = self.setPoint(obstacle.pose.position.x, obstacle.pose.position.y, m)
                    obstacle.pose.position.x = obstacle.pose.position.x + random_item[index_x]
                    obstacle.pose.position.y = obstacle.pose.position.y + random_item[index_y]
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)      

def main():
    rospy.init_node('combination_obstacle_3')
    try:
        combination = Combination()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
