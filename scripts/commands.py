#!/usr/bin/python
########################
# Copywrite Zachary Kratochvil 2022
# All rights reserved.
########################

import time

import rospy
from speech_recognition_msgs.msg import SpeechRecognitionCandidates
from pet.msg import LegAngle

class VoiceCommand:
    def __init__(self):
        self.leg_pub = rospy.Publisher("legs",
                                        LegAngle, queue_size=10)

        rospy.init_node('voice_commands', anonymous=False)

    def start(self):
        rospy.Subscriber("speech_to_text", SpeechRecognitionCandidates, self.process_speech, queue_size=10)
        rospy.spin()

    def process_speech(self, data):
        speech = "".join(data.transcript)

        if "sit down" in speech or "sit up" in speech or speech == "sit":
            self.leg_pub.publish(180, 180, 0, 0)
        elif "stand up" in speech or speech == "up" or speech == "stand":
            self.leg_pub.publish(180, 180, 180, 180)
        elif "lie down" in speech or speech == "down":
            self.leg_pub.publish(0, 0, 0, 0)
        elif "hello" in speech:
            self.leg_pub.publish(180, 150, 150, 90)

            time.sleep(.5)
            for i in range(6):
                self.leg_pub.publish(0, 150, 150, 90)
                time.sleep(.2)
                self.leg_pub.publish(120, 150, 150, 90)
                time.sleep(.2)
            self.leg_pub.publish(180, 180, 180, 180)

if __name__ == "__main__":
    vc = VoiceCommand()
    vc.start()