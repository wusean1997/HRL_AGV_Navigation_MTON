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
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
from respawnSubGoal import Respawn_subgoal


class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.get_subgoalbox = False
        self.position = Pose()
        self.subgoal_x = 0
        self.subgoal_y = 0
        self.past_distance = 0.0
        self.past_heading = 0.0
        self.past_x = 0.0
        self.past_y = 0.0
        self.angle = 0.0
        self.current_distance = 0.0
        self.current_heading = 0.0
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.respawn_subgoal = Respawn_subgoal()

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        subgoal_angle = math.atan2(self.subgoal_y - self.position.y, self.subgoal_x - self.position.x)
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = subgoal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.angle = round(goal_angle, 2)
        self.heading = round(heading, 2)

    def LaserFilter(self, scan):
        scan_range = []
        for i in range(len(scan)):
            if scan[i] > 1:
                scan_range.append(1.0)
            else:
                scan_range.append(scan[i])

        return scan_range

    def GetLaser(self, scan):
        scan_range = []
        # for i in range(len(scan.ranges)):
        for i in range(0,360, 15):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
                
        return scan_range

    def Normalize(self, data, Max, min, num):
        if isinstance(data,list):
            nor_data = [round((data[i]-min)/(Max-min), num) for i in range(len(data))]
        else:
            nor_data = round((data-min)/(Max-min), num)

        return nor_data
    
    def posState(self, x, y):
        if y >= -2.5 :
            case = "A"
        else:
            case = "B"
            
        return case

    def subGoal(self, action):
        goal_x = 0
        goal_y = 0 
        if action == 0:
            radius = 0.0#m
            angular = 0.0#rad
        else:
            radius = 1.5#m
            angular = self.heading + (action - 4) * (pi / 4.0)

        goal_x = self.position.x + radius * math.cos(angular)
        goal_y = self.position.y + radius * math.sin(angular)

        return goal_x, goal_y

    def getState(self, scan):
        angle = self.heading
        min_range = 0.208

        laser = self.GetLaser(scan)
        self.current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)

        laser = self.Normalize(laser, 3.5, 0.0, 2)
        angle = self.Normalize(angle, pi, -pi, 2)
        distance = self.Normalize(self.current_distance, 10.0, 0.0, 2)
        self.pub_cmd_vel.publish(Twist())

        return laser + [angle, distance]

    def getsubState(self, scan):
        min_range = 0.208
        done = False

        laser = self.GetLaser(scan)
        laser = self.LaserFilter(laser)

        if min_range > min(laser) > 0 or self.position.x >= 8 or self.position.x <= -2 or self.position.y >= -1 or self.position.y <= -5:
            done = True
            self.pub_cmd_vel.publish(Twist())
        
        current_distance = round(math.hypot(self.subgoal_x - self.position.x, self.subgoal_y - self.position.y),2)

        if current_distance < 0.1:
            self.get_subgoalbox = True
            self.pub_cmd_vel.publish(Twist())

        distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)

        if distance < 0.2:
            self.get_goalbox = True
            self.pub_cmd_vel.publish(Twist())
        
        laser = self.Normalize(laser, 1.0, 0.0, 2) 
        heading = self.Normalize(self.heading, pi, -pi, 3) 
        distance = self.Normalize(current_distance, 5.0, 0.0, 3)

        return laser + [heading, distance], done, self.get_subgoalbox, self.get_goalbox

    def setReward(self, state, done):
        distance = self.current_distance
        heading = self.current_heading

        diff_distance = round((self.past_distance - self.current_distance),3)
        x = self.position.x
        y = self.position.y
        past_x = self.past_x
        past_y = self.past_y

        if done:
            rospy.loginfo("Collision!!")
            reward = -550
            self.pub_cmd_vel.publish(Twist())
            try:
                self.reset_proxy()
            except (rospy.ServiceException) as e:
                print("gazebo/reset_simulation service call failed")

        elif self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 500
            self.pub_cmd_vel.publish(Twist())
            self.get_goalbox = False

        else:
            """ Entry Reward """
            past_case = self.posState(past_x, past_y)
            case = self.posState(x, y)
            n = 50
            m = -10

            if past_case == "A" and case == "A":
                if (x - past_x) > 0:
                    reward = (x - past_x) * n
                else:
                    reward = m + (x - past_x) * n
            elif past_case == "B" and case == "B":
                if (diff_distance) > 0:
                    reward = (diff_distance) * n
                else:
                    reward = m + (diff_distance) * n
            else:
                rospy.loginfo("Over region!!")
                reward = -100

        self.past_x = self.position.x
        self.past_y = self.position.y
        self.past_heading = heading
        self.past_distance = distance

        return reward	

    def step(self, done):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state = self.getState(data)
        reward = self.setReward(state, done)

        return np.asarray(state), reward

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        self.goal_x, self.goal_y = self.respawn_goal.getPosition()
        """
        if initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
        """
        self.goal_distance = self.getGoalDistace()
        state = self.getState(data)
        self.past_x = self.position.x
        self.past_y = self.position.y

        return np.asarray(state), round(self.goal_x, 2), round(self.goal_y, 2)
    
    def action_step(self, action):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done, get, getgoal = self.getsubState(data)
        self.get_subgoalbox = False
        

        return np.asarray(state), done, get, getgoal

    def action_reset(self, action):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        
        self.subgoal_x, self.subgoal_y = self.subGoal(action)
        """
        if initSubgoal:
            self.respawn_subgoal.getPosition(self.subgoal_x, self.subgoal_y)
        else:
            self.respawn_subgoal.getPosition(self.subgoal_x, self.subgoal_y, delete=True)
        """
        self.respawn_subgoal.getPosition(self.subgoal_x, self.subgoal_y)

        state, done, get, getgoal = self.getsubState(data)

        return np.asarray(state), round(self.subgoal_x, 2), round(self.subgoal_y, 2)