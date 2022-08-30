#!/usr/bin/python
########################
# Copywrite Zachary Kratochvil 2022
# All rights reserved.
########################

import numpy as np
import time
import threading

import rospy
from speech_recognition_msgs.msg import SpeechRecognitionCandidates
from pet.msg import LegAngle
from std_msgs.msg import ColorRGBA

class VoiceCommand:
    def __init__(self):
        self.leg_pub = rospy.Publisher("legs",
                                        LegAngle, queue_size=10)
        self.led_pub = rospy.Publisher("status_led", ColorRGBA, queue_size=10)

        rospy.init_node('voice_commands', anonymous=False)

    def start(self):
        rospy.Subscriber("speech_to_text", SpeechRecognitionCandidates, self.process_speech, queue_size=10)
        rospy.spin()

    def process_speech(self, data):
        speech = "".join(data.transcript)

        if "k dog" not in speech:
            self.led_fail()
            return

        pose_vector = [
            "lie" in speech,
            "sit" in speech,
            "stand" in speech
            ]

        transition_vector = [
            "down" in speech,
            "up" in speech
        ]

        action_vector = [
            "hello" in speech,
            "hi" in speech
        ]

        # handle direct pose commands
        if np.sum(pose_vector) > 1:
            self.led_fail()
            return
        if np.sum(pose_vector) == 1 and np.sum(action_vector) > 0:
            self.led_fail()
            return
        elif np.sum(pose_vector) == 1:
            self.make_pose(pose_vector)
            self.led_success()
            return

        if np.sum(transition_vector) > 1:
            self.led_fail()
            return
        # handle "down"
        elif transition_vector[0] == 1:
            self.make_pose([1,0,0])
            self.led_success()
            return
        # handle "up"
        elif transition_vector[1] == 1:
            self.make_pose([0,0,1])
            self.led_success()
            return

        # handle hello action
        if np.sum(action_vector) > 0:
            self.led_success()
            self.leg_pub.publish(180, 150, 150, 90)

            time.sleep(.5)
            for i in range(6):
                self.leg_pub.publish(0, 150, 150, 90)
                time.sleep(.2)
                self.leg_pub.publish(120, 150, 150, 90)
                time.sleep(.2)
            self.leg_pub.publish(180, 180, 180, 180)
            return

        else:
            self.led_fail()

    def make_pose(self, pose_vector):
        # lie down
        if pose_vector[0]:
            self.leg_pub.publish(0, 0, 0, 0)
        # sit
        elif pose_vector[1]:
            self.leg_pub.publish(180, 180, 0, 0)
        # stand up
        elif pose_vector[2]:
            self.leg_pub.publish(180, 180, 180, 180)

    def led_success(self):
        self.led_pub.publish(0,1,0,1)
        #threading.timer(1,self.led_clear).star()

    def led_fail(self):
        self.led_pub.publish(1,1,0,1)
        #threading.timer(1,self.led_clear).star()

    def led_clear(self):
        self.led_pub.publish(0,0,0,0)



if __name__ == "__main__":
    vc = VoiceCommand()
    vc.start()