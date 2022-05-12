#!/usr/bin/env python
# Authors: Eric
# Update: 2021/07/11
import rospy
import random
import time
import math
import os
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('base_action/src',
                                                'base_action/subgoal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = 1.2
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.index = 0
        self.init = True
        self.n = 0

    def checkModel(self, model):
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.init = False


    def respawnModel(self):
        if self.init:
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
            #rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                            #self.goal_position.position.y)
            self.init = False
        else:
            self.deleteModel()
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
            #rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                            #self.goal_position.position.y)

    def deleteModel(self):
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox(self.modelName)          

    def getPosition(self, position_check=False, delete=False):       
        if self.stage == 0:
            while position_check:
                goal_x = random.randrange(-15, -5) / 10.0
                goal_y = random.randrange(-40, -30) / 10.0
                position_check = False
                
                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        elif self.stage == 1:
            while position_check:
                distance = [1.0, 2.0]
                list_r = self.n % 2
                list_a = self.n % 16
                goal_radius = distance[list_r]#m
                #goal_angle = math.pi * abs(self.n / 100) / 4.0
                goal_angle = math.pi * list_a / 8.0
                goal_x = round(goal_radius * math.cos(goal_angle),3)
                goal_y = round(goal_radius * math.sin(goal_angle),3)
                position_check = False

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
        self.n += 1       
        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
