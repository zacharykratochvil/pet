#!/usr/bin/python
########################
# Copywrite Zachary Kratochvil 2022
# All rights reserved.
########################

import time
import threading
import rospy
import RPi.GPIO as GPIO
import signal
import sys

from pet.msg import LegAngle

class Enable:
    def __init__(self):
        rospy.init_node('enable', anonymous=False)

        self.leg_pub = rospy.Publisher("legs",
                                        LegAngle, queue_size=10)
        
        self.leg_pub.publish(90,90,90,90)

        self.timer = None
        signal.signal(signal.SIGINT, self.shutdown)

    def shutdown(self, sig, frame):
        try:
            self.timer.cancel()
        finally:
            GPIO.cleanup()

    def start(self):
        self.timer = threading.Timer(1, self.set_low)
        self.timer.start() 

    def set_low(self):
        GPIO.output(6, GPIO.LOW)
        self.start()

if __name__ == "__main__":
    e = Enable()
    time.sleep(.1)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(6, GPIO.OUT)

    e.start()


