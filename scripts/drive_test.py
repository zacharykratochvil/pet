#!/usr/bin/python
########################
# Copywrite Zachary Kratochvil 2022
# All rights reserved.
########################

import threading
import rospy
import numpy as np
import sklearn.cluster as sklc

from slam.common.particle_filter import ParticleFilter

from freenove_ros.msg import TwistDuration
from sensor_msgs.msg import PointCloud
from pet.msg import Localization

class Test:
    def __init__(self):
        rospy.init_node('drive_test', anonymous=True)

        self.drive_pub = rospy.Publisher("drive_twist",
                                        TwistDuration, queue_size=1)

        self.map_sub = rospy.Subscriber("map", PointCloud, self.map, queue_size=1)
        self.robot_sub = rospy.Subscriber("robot_location", Localization, self.robot, queue_size=1)
        #self.particle_sub = rospy.Subscriber("robot_particles", PointCloud, self.robot, queue_size=1)
        self.map = None
        self.robot = None

    def start(self):
        self.t = threading.Timer(5, self.move)
        self.t.start()

        rospy.spin()

    def map(self, data):
        self.map = ParticleFilter.decloud(data)

    def robot(self, data):
        #self.robot = ParticleFilter.decloud(data)
        self.robot = data.candidates
    
    def move(self):
        self.t.cancel()
        self.t = threading.Timer(5, self.move)
        self.t.start()

        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 0
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 30
        message.duration = .33
        self.drive_pub.publish(message)

    def move2(self):
        self.t.cancel()
        self.t = threading.Timer(5, self.move)
        self.t.start()

        # find largest cluster in robot map
        '''
        ms = sklc.MeanShift(bandwidth=.5, bin_seeding=True, min_bin_freq=3, n_jobs=1, max_iter=50)
        ms.fit(self.robot[:,:2])
        values, counts = np.unique(ms.labels_, return_counts=True)
        counts = np.array(counts, float)
        counts[values==-1] = np.nan
        max_ind = np.argmax(counts)
        robot_coords = ms.cluster_centers_[values[max_ind],:]
        rospy.logerr(robot_coords)
        '''
        if self.robot[0].confidence < .6 or self.robot[0].spread > .4:
            return
        else:
            robot_coords = (self.robot[0].position.x, self.robot[0].position.y)

        # navigate from cluster center to origin
        origin_vector = np.asarray([0,1])
        theta_to_origin = np.arctan2(-robot_coords[1],-robot_coords[0]) - np.arctan2(origin_vector[1],origin_vector[0]) #arctan2 is opposite over adjacent or (y,x)
        theta_to_origin = 180/np.pi*theta_to_origin
        if theta_to_origin > 180:
            theta_to_origin = -(360 - theta_to_origin)
        elif theta_to_origin > 360:
            rospy.logerr("invalid angle calculation of robot position relative to origin")
        #theta_robot = np.mean(self.robot[:,2])
        theta_robot = self.robot[0].orientation.z
        rospy.logerr(f"origin {theta_to_origin}")
        rospy.logerr(f"robot {robot_coords}; angle {theta_robot}")
        turn_angle = (theta_to_origin-theta_robot) % 360
        if turn_angle > 180:
            turn_angle = -(360 - turn_angle)
        rospy.logerr(f"turn {turn_angle}")
        turn_gain = .2*min(1,np.linalg.norm(robot_coords))

        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 0
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = turn_gain*turn_angle
        message.duration = .33
        self.drive_pub.publish(message)


    def m2a0(self):
        rospy.logerr("success")
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = -.9
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 0
        message.duration = 3
        self.drive_pub.publish(message)

    def m1a45(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 1
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 45
        message.duration = 1
        self.drive_pub.publish(message)
        t1 = threading.Timer(1, test.m1a45)
        t1.start()

    def m22an0(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 2
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 0
        message.duration = 1
        self.drive_pub.publish(message)        

    def m0an90(self):
        message = TwistDuration()
        message.velocity.linear.x = 0
        message.velocity.linear.y = 2
        message.velocity.linear.z = 0
        message.velocity.angular.x = 0
        message.velocity.angular.y = 0
        message.velocity.angular.z = 0
        message.duration = 1
        self.drive_pub.publish(message)

if __name__ == "__main__":
    test = Test()
    
    t = threading.Timer(2, test.m2a0)
    #t.start()

    '''
    t1 = threading.Timer(4, test.m1a45)
    #t1.start()

    t2 = threading.Timer(6, test.m22an0)
    #t2.start()
    '''

    test.start()