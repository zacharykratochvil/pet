#!/usr/bin/python
########################
# Copywrite Zachary Kratochvil 2022
# All rights reserved.
########################

import rospy
from pet.msg import LegAngle
from freenove_ros.msg import ServoAngle

class LegController:
    def __init__(self):
        self.servo_pub = rospy.Publisher("servos",
                                        ServoAngle, queue_size=10)

        # initialize node
        rospy.init_node("leg_actuator", anonymous=False)

    ###
    # Set up subscriber
    ###
    def start(self):
        rospy.Subscriber("legs", LegAngle, self.set_angle, queue_size=10)
        rospy.spin()

    ###
    # Activate servos when we get a message
    ###
    def set_angle(self, data):

        def validate(number):
            if not (0 <= number <= 180):
                raise Exception("Invalid servo angle.")
        
        validate(data.front_left_angle)
        data.front_left_angle = min(170,max(10,data.front_left_angle))
        validate(data.front_right_angle)
        data.front_right_angle = min(170,max(10,data.front_right_angle))
        validate(data.back_left_angle)
        data.back_left_angle = min(160,max(30,data.back_left_angle))
        validate(data.back_right_angle)
        data.back_right_angle = min(160,max(30,data.back_right_angle))
                
        self.servo_pub.publish(id=3, angle=180-data.front_left_angle)
        self.servo_pub.publish(id=4, angle=data.front_right_angle)
        self.servo_pub.publish(id=5, angle=180-data.back_left_angle)
        self.servo_pub.publish(id=6, angle=data.back_right_angle)


if __name__ == "__main__":
    lc = LegController()
    lc.start()