#!/usr/bin/python
########################
# Copywrite Zachary Kratochvil 2022
# All rights reserved.
########################

import threading
import rospy

from freenove_ros.msg import TwistDuration

class Test:
    def __init__(self):
        rospy.init_node('drive_test', anonymous=False)

        self.drive_pub = rospy.Publisher("drive_twist",
                                        TwistDuration, queue_size=10)
    
    def m2a0(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = -2
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 0
        message.duration = 1
        self.drive_pub.publish(message)

    def m1a45(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 1
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 45
        message.duration = 2
        self.drive_pub.publish(message)

    def m22an0(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 2
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 0
        message.duration = 2
        self.drive_pub.publish(message)        

    def m0an90(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 2
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 0
        message.duration = 2
        self.drive_pub.publish(message)

if __name__ == "__main__":
    test = Test()
    t = threading.Timer(2, test.m2a0)
    t.start()

    t1 = threading.Timer(4, test.m1a45)
    t1.start()

    t2 = threading.Timer(6, test.m22an0)
    t2.start()