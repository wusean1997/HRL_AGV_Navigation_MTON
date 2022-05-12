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
import random
import time
import os
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose

class Respawn_subgoal():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('area_dqn/src',
                                                'area_dqn/subgoal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.modelName = 'subgoal'
        self.subgoal_position = Pose()
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.init = True

    def checkModel(self, model):
        for i in range(len(model.name)):
            if model.name[i] == "subgoal":
                self.init = False

    def respawnModel(self):
        if self.init:
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.subgoal_position, "world")
            #rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)
            self.init = False
        else:
            self.deleteModel()
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.subgoal_position, "world")
            #rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)

    def deleteModel(self):
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox(self.modelName)



    def getPosition(self, subgoal_x, subgoal_y):
        self.subgoal_position.position.x = subgoal_x
        self.subgoal_position.position.y = subgoal_y
        self.respawnModel()
        time.sleep(0.5)