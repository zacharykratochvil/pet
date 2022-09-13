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
        
    def m1a30(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 1
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 30
        message.duration = 1
        self.drive_pub.publish(message)

    def m0an90(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 0
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = -90
        message.duration = 1
        self.drive_pub.publish(message)        

if __name__ == "__main__":
    test = Test()

    #t1 = threading.Timer(15, test.m1a30)
    #t1.start()

    #t2 = threading.Timer(30, test.m0an90)
    #t2.start()