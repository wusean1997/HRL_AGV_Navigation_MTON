#! /usr/bin/env python
#coding=utf-8
import rospy
import math
import time
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from matplotlib.patches import Rectangle, Arrow

def cmd_vel_callback(msg):
    global dx
    global dy
    lin = msg.linear.x
    ang = msg.angular.z
    dx = lin * math.cos(ang)
    dy = lin * math.sin(ang)
    #print(u,v)

plt.ion()
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.plot([], [], 'bo')
ax.set_xlim(-0.52,0.52)
ax.set_ylim(-0.52,0.52)

dx, dy = 1.0, 1.0
arrow = Arrow(0, 0, dx, dy, width=0.05)

a = ax.add_patch(arrow)
plt.draw()

rospy.init_node('cmd_vel_visual_node')

while True:
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
    a.remove()
    arrow = Arrow(0, 0, dx, dy, width=0.05)
    a = ax.add_patch(arrow)

    fig.canvas.draw_idle()
    plt.pause(0.2)
