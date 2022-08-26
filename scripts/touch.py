#!/usr/bin/python
########################
# Copywrite Zachary Kratochvil 2022
# All rights reserved.
########################

import time
import rospy
import RPi.GPIO as GPIO
import threading

from pet.msg import LegAngle

class TouchCommand:
    def __init__(self):
        self.leg_pub = rospy.Publisher("legs",
                                        LegAngle, queue_size=10)

        rospy.init_node('touch_commands', anonymous=False)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(10, GPIO.IN)
        GPIO.setup(11, GPIO.IN)

    def start(self):
        t = threading.Timer(.05, self.check)
        t.start()

    def check(self):
        left = not GPIO.input(10)
        right = not GPIO.input(11)

        if left:
            self.leg_pub.publish(0,80,80,80)
        elif right:
            self.leg_pub.publish(80,0,80,80)
        else:
            self.leg_pub.publish(80,80,80,80)

        self.start()


if __name__ == "__main__":
    tc = TouchCommand()
    tc.start()