#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped, Twist

class DockingController:
    def __init__(self):
        rospy.init_node('docking_controller', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('/aruco_pose', PoseStamped, self.pose_callback)

    def pose_callback(self, msg):
        target_x = msg.pose.position.x
        target_z = msg.pose.position.z

        cmd = Twist()
        if target_z > 0.2:
            cmd.linear.x = 0.1
        else:
            cmd.linear.x = 0.0

        if abs(target_x) > 0.05:
            cmd.angular.z = -0.2 * target_x
        else:
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    DockingController()
    rospy.spin()
