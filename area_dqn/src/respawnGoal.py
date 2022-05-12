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
"""
import rospy
import random
import time
import os
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose
class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('area_dqn/src',
                                                'area_dqn/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = -1.0
        self.init_goal_y = -3.5
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.check_model = False
        self.index = 0
        self.i = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True
                self.i = i

    def respawnModel(self):
        if self.check_model:
            obstacle = ModelState()
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            if model.name[self.i] == 'goal':
                obstacle.model_name = 'goal'
                obstacle.pose = model.pose[self.i]
                obstacle.pose.position.x = self.goal_position.position.x
                obstacle.pose.position.y = self.goal_position.position.y
                self.pub_model.publish(obstacle)
        else:
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
            #rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)

    def deleteModel(self):
        if self.check_model:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox(self.modelName)


    def getPosition(self):
        if rospy.is_shutdown():
            self.deleteModel()

        goal_x = -1.5
        goal_y = -4.5 + random.randrange(0,10) / 10.0
        position_check = False
                
        self.goal_position.position.x = goal_x
        self.goal_position.position.y = goal_y

        time.sleep(0.5)
        self.respawnModel()

        return self.goal_position.position.x, self.goal_position.position.y
"""
import rospy
import random
import time
import os
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('area_dqn/src',
                                                'area_dqn/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = -1.0
        self.init_goal_y = -3.5
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.index = 0
        self.init = True

    def checkModel(self, model):
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.init = False

    def respawnModel(self):
        if self.init:
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
            #rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)
            self.init = False
        else:
            self.deleteModel()
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
            #rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)

    def deleteModel(self):
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox(self.modelName)


    def getPosition(self):
        goal_x = -1.0 + random.randrange(0,10) / 10.0
        goal_y = -3.5 - random.randrange(0,10) / 10.0
        position_check = False
                
        self.goal_position.position.x = goal_x
        self.goal_position.position.y = goal_y

        time.sleep(0.5)
        self.respawnModel()

        return self.goal_position.position.x, self.goal_position.position.y

"""
import rospy
import random
import time
import os
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('area_dqn/src',
                                                'area_dqn/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = -1.0
        self.init_goal_y = -3.5
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()
        
        if self.stage == 0:
            while position_check:
                goal_radius = random.randrange(50,100) / 100.0
                goal_angle = random.randrange(-314,314) / 100.0
                goal_x = round(goal_radius * math.cos(goal_angle),3)
                goal_y = round(goal_radius * math.sin(goal_angle),3)
                position_check = False

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        elif self.stage == 1:
            while position_check:
                goal_x = -1.0 + random.randrange(0,10) / 10.0
                goal_y = -3.5 - random.randrange(0,10) / 10.0
                position_check = False
                
                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y


        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
"""