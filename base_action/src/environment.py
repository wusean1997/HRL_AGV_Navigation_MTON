#!/usr/bin/env python
# Authors: Morrison
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
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as transforms
import roslaunch
toGRAY = False

class Env():
    def __init__(self):
        # initiliaze
        #rospy.init_node('GazeboWorld', anonymous=False)
        #-----------Publisher and Subscriber-------------
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_goal_heading_odom = rospy.Subscriber('odom2', Odometry, self.getGoalAngle)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        #------------Params--------------------
        self.useLaser = True
        self.show_cam = False
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.get_goalbox = False
        self.position = Pose()
        self.init_robot_x = 0.0
        self.init_robot_y = 0.0
        self.respawn_goal = Respawn()
        self.past_x = 0.0
        self.past_y = 0.0
        self.stuck = 0
        self.past_distance = 0.
        self.past_heading = 0.
        self.current_distance = 0.
        self.current_heading = 0.
        self.goal_heading = 0.
        self.current_minScanrange = 0.
        self._num_checkpoints = 50
        self._checkpoint_reward = 0.002
        self.check_reward = 0
        self.treshold = 0.
        self.viscount = 0.
        self.time_punishment=0.
        self.n = 0
        self.past_done = False
        self.n = 0
        self.init = True
        self.set = False
        #self.past_laser = [[],[],[]]
        
        rospy.sleep(2.)
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)
        # self.launch = roslaunch.parent.ROSLaunchParent(uuid, [path + '/launch/nav_gazebo.launch'])

    def shutdown(self):
        # Stop robot by publishing an empty Twist
        #rospy.loginfo("Stop Moving")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 5)

        return goal_distance

    def getGoalAngle(self, odom2):
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        # roll, pitch, yaw
        _, _, yaw = euler_from_quaternion(orientation_list)
        goal_angle = math.atan2(self.goal_y - position.y, self.goal_x - position.x)
        

        goal_heading = goal_angle - yaw
        if goal_heading > pi:
            goal_heading -= 2 * pi
        elif goal_heading < -pi:
            goal_heading += 2 * pi
        self.goal_heading = goal_heading

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        # roll, pitch, yaw
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)
        self.yaw = yaw
        self.goal_angle = goal_angle

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
        scan_optima = []
        """
        # for i in range(len(scan.ranges)):
        for i in range(0,360, 15):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        """
        for i in range(360):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(round(scan.ranges[i],2))

        for i in range(360/15):
            scan_optima.append(min(scan_range[i:i+15]))

        return scan_optima

    def Normalize(self, data, Max, Min, num):
        if isinstance(data,list):
            nor_data = []
            for i in range(len(data)):
                value = round((data[i]-Min)/(Max-Min), num)
                if value >= 1.0:
                    nor_data.append(1.0)
                elif value <= 0.0:
                    nor_data.append(0.0)
                else:
                    nor_data.append(value)
        else:
            nor_data = 0.0
            value = round((data-Min)/(Max-Min), num)
            if value >= 1:
                nor_data = 1.0
            elif value <= 0:
                nor_data = 0.0
            else:
                nor_data = value

        return nor_data

    def Diff_heading(self, past_heading, heading):
        diff = abs(past_heading - heading)
        if diff > pi:
            diff = 2 * pi - diff
        
        diff = self.Normalize(diff, pi, 0)

        return diff
        
    def getStatebyScan(self, scan, past_action):
        #heading = self.heading
        #yaw = self.yaw
        #goal_angle = self.goal_angle
        # min_range = 0.16 #burger
        min_range = 0.208 #waffle
        done = False

        laser = self.GetLaser(scan)
        self.current_minScanrange = min(laser)

        self.current_distance = self.getGoalDistace()
        if min_range > min(laser) > 0:
            done = True
            self.stuck += 1
 
        if self.current_distance < 0.05:
            self.get_goalbox = True
        
        laser = self.LaserFilter(laser)       
        self.current_heading = self.heading
        
        if self.init:
            laser_difference = [laser[i] - laser[i] for i in range(len(laser))]
            self.init = False
        else:
            laser_difference = [self.last_laser[i] - laser[i] for i in range(len(laser))]

        self.last_laser = laser    
        laser = self.Normalize(laser, 1.0, 0.0, 1)#2
        laser_difference = self.Normalize(laser_difference, 0.5, -0.5, 1)#2
        #past_action[0] = self.Normalize(past_action[0], 0.52, 0.0, 2)
        #past_action[1] = self.Normalize(past_action[1], 2.79, -2.79, 2)
        #goal_angle = self.Normalize(goal_angle, pi, -pi)
        #yaw = self.Normalize(yaw, pi, -pi) 
        current_heading = self.Normalize(self.current_heading, pi, -pi, 1)#2
        current_distance = self.Normalize(self.current_distance, 5.0, 0.0, 2)#3
        
        """
        for pa in past_action:
            laser.append(pa) # laser[-2]:linear velocity , laser[-1]:angle velocity
        data = laser + [goal_angle, yaw, current_heading, current_distance]
        #print(data)
        return laser + [goal_angle, yaw, current_heading, current_distance], done
        """
        return laser + laser_difference + [current_heading, current_distance], done
        """
        past = self.past_laser
        
        if self.init == 3:   
            data = laser + laser + laser + laser + [current_heading, current_distance]    
            past[0] = laser
            self.init = 2

        elif self.init == 2:
            data =  past[0] + laser + laser + laser + [current_heading, current_distance]
            past[1] = laser
            self.init = 1
        
        elif self.init == 1:
            data =  past[0] + past[1] + laser + laser + [current_heading, current_distance]
            past[2] = laser
            self.init = 0

        else:
            data =  past[0] + past[1] + past[2] + laser + [current_heading, current_distance]
            past[0] = past[1]
            past[1] = past[2]
            past[2] = laser
        
        return data, done"""
        

    def GetReward(self, state, done):
        """
        #Test
        reward = [0,0,0,0]
        #n = int(self.n / 100)

        distance = math.hypot(self.position.y, self.position.x)
        region_yaw = abs(math.atan2(self.position.y, self.position.x))
        theata = abs(math.atan2(self.goal_y, self.goal_x))

        max_yaw = theata+(math.pi/4)
        min_yaw = theata-(math.pi/4)

        if region_yaw >= min_yaw and region_yaw <= max_yaw:
            if distance >= 0.5:
                reward[0] = 1
                if distance >= 1.0:
                    reward[1] = 1
                    if distance >= 1.25:
                        reward[2] = 1
                        if distance >= 1.5:
                            reward[3] = 1

        #print("reward:" + str(reward))
        #print("n:" + str(n) + "theata:" + str(theata) + "region_yaw:" + str(region_yaw))

        #self.n += 1

        return reward

        """
        #Train
        #current_heading = self.current_heading#state[-2]
        current_distance = self.current_distance#state[-1]
        if self.set:
            self.past_distance = round(math.hypot(self.goal_x - self.init_robot_x, self.goal_y - self.init_robot_y), 5)
            self.set = False
        distance_rate = (self.past_distance - current_distance) 

        if done:
            rospy.loginfo("Collision!!")
            reward = -110.  

            while self.stuck >= 5:
                self.pub_cmd_vel.publish(Twist())
                rospy.wait_for_service('gazebo/reset_simulation')
                try:
                    self.reset_proxy()
                except (rospy.ServiceException) as e:
                    print("gazebo/reset_simulation service call failed")
                self.stuck = 0
                self.init = True
                self.set = True
    
        else:
            self.stuck = 0
            if self.get_goalbox:
                rospy.loginfo("Goal!!")
                self.n += 1
                reward = 500.
                self.get_goalbox = False

            else:
                if distance_rate > 0:
                    reward = 200.* distance_rate
                else:
                    reward = -10. + 200.* distance_rate

        #self.past_heading = current_heading  
        self.past_distance = current_distance
        self.past_done = done
        #self.past_x = self.position.x
        #self.past_y = self.position.y

        #print("\033[1;37mDistance: " + str(diff_distance) + ", heading:" + str(diff_heading)  +"\033[0m" )
        print("\033[1;37mReward: " + str(reward) + ", Times:" + str(self.n)  +"\033[0m" )
        return round(reward,3)
        # *********************************************************#

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)   
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        state, done  = self.getStatebyScan(data, past_action)
        reward = self.GetReward(state, done)

        return np.asarray(state), reward, done       

    def reset(self, init):
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

        if init:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        #self.goal_distance = self.getGoalDistace()
        self.check_reward = 0
        self.time_punishment=0.
        self.n = 0
        self.init = True
        self.init_robot_x = self.position.x
        self.init_robot_y = self.position.y
        self.past_distance = self.getGoalDistace()
        state, done = self.getStatebyScan(data, [0.,0.])
        #self.past_distance = self.current_distance
        #self.past_heading = self.current_heading
        #self.past_x = self.position.x
        #self.past_y = self.position.y


        return np.asarray(state)
