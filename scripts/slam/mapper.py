#!/usr/bin/python

import common.particle_filter as pf
import rospy
import time
import numpy as np
import threading

from sensor_msgs.msg import Imu, PointCloud
from geometry_msgs.msg import Point32

class Mapper(pf.ParticleFilter):
    def __init__(self, options=pf.FilterOptions()):

        options.options["initial_linear_dist"] = pf.UniformDistribution2D((-10,10),(-10,10))
        options.options["null_linear_dist"] = pf.UniformDistribution2D((-10,10),(-10,10))
        options.options["initial_weight"] = 1/time.time()
        super().__init__(options)

        self.robot_particles = None

        self.map_timer = threading.Timer(self.options["publish_interval"], self.publish)

        self.map_pub = rospy.Publisher("map", PointCloud, queue_size=2)

        self.measure_count = 0
        self.measurement_sub = rospy.Subscriber("measured_particles", PointCloud, self.map, queue_size=1)
        self.robot_sub = rospy.Subscriber("robot_particles", PointCloud, self.update_robot, queue_size=1)

    def start(self):

        rospy.init_node("robot_localizer", anonymous=False)

        #self.map_timer.start()
        
        rospy.spin()

    def map(self, data):

        if self.locked == True or type(self.robot_particles) is type(None):
            return False
        else:
            self.locked = True

        measured_particles = self.decloud(data)
        if len(measured_particles) == 0:
            self.locked = False
            return True
    
        new_particles = np.empty(np.shape(measured_particles))
        re_weights = set()

        for measured_i in range(np.shape(measured_particles)[0]):
            
            # add new particle measured distance from a random reference particle
            ref_index = np.random.randint(len(self.robot_particles))
            ref_particle = self.robot_particles[ref_index,:]

            new_particles[measured_i, :] = self.transform_one(measured_particles[measured_i,:], ref_particle)
            new_particles[measured_i, self.WEIGHT] = 1/time.time()

            # downweight particles on path from reference to new particle
            #re_origin_particles = self.particles[:,self.X:self.Y+1] - new_particles[measured_i,self.X:self.Y+1]
            magnitude = np.sqrt(np.sum(measured_particles[measured_i,self.X:self.Y+1]**2))
            measured_angle = 180/np.pi*np.arctan2(measured_particles[measured_i,self.Y],measured_particles[measured_i,self.X])
            angle = (measured_angle + ref_particle[self.ANGLE] + 180) % 360 - 180
            delta_angle = 15 #degrees
            
            reweighted_count = 0
            base_angles = 180/np.pi*np.arctan2(self.particles[:,self.Y],self.particles[:,self.X])
            for map_particles_i in range(len(self.particles)):
                test_magnitude = np.sqrt(np.sum(self.particles[map_particles_i,self.X:self.Y+1]**2))
                test_angle = (base_angles[map_particles_i] - angle + 540) % 360 - 540

                if test_magnitude < magnitude and -delta_angle < test_angle and test_angle < delta_angle:
                    re_weights.add(map_particles_i)
                    reweighted_count += 1
            
        indecies = list(re_weights)
        self.particles[indecies,self.WEIGHT] = self.particles[indecies,self.WEIGHT]/2
        rospy.loginfo(reweighted_count)

        self.particles = np.vstack([self.particles, new_particles])

        self.locked = False
        if self.measure_count % 10 == 0:
            self.measure_count = 1
            self.publish()
            rospy.loginfo("resampling")
            return self.resample()
        else:
            self.measure_count += 1
            rospy.loginfo("done")
            return True

    def update_robot(self, data):
        self.robot_particles = self.decloud(data)

    def publish(self):
        # restart timer
        self.map_timer.cancel()
        self.map_timer = threading.Timer(self.options["publish_interval"], self.publish)
        #self.map_timer.start()
        
        self.map_pub.publish(self.make_cloud(self.particles))


if __name__ == "__main__":
    
    options = pf.FilterOptions({"publish_interval": 1e6, "resample_interval": 1e6})
    map = Mapper(options)
    map.start()

