#!/usr/bin/python

import common.particle_filter as pf
import rospy
import time
import numpy as np
import threading
import random
from scipy.special import expit as sigmoid

from sensor_msgs.msg import Imu, PointCloud
from geometry_msgs.msg import Point32

class Mapper(pf.ParticleFilter):

    #map_count = 0

    def __init__(self, options=pf.FilterOptions()):

        self.start_time = time.time()

        options.options["initial_linear_dist"] = pf.UniformDistribution2D((-10,10),(-10,10))
        options.options["null_linear_dist"] = pf.UniformDistribution2D((-10,10),(-10,10))
        options.options["initial_weight"] = 0
        super().__init__(options)

        self.robot_particles = None
    
        self.prune_timer = threading.Timer(-1,lambda x: x)
        self.grow_timer = threading.Timer(-1,lambda x: x)
        self.map_timer = threading.Timer(self.options["publish_interval"], self.publish)
        
        self.map_pub = rospy.Publisher("map", PointCloud, queue_size=2)
        self.robot_sub = rospy.Subscriber("robot_particles", PointCloud, self.update_robot, queue_size=2)
        self.robot_map_sub = rospy.Subscriber("local_map_particles", PointCloud, self.grow_map, queue_size=2)
        self.measurement_sub = rospy.Subscriber("measured_particles", PointCloud, self.prune_map, queue_size=2)
        '''else:
            self.robot_particles = np.zeros([1,4])
            #self.map_pub = rospy.Publisher("map"+str(self.map_count), PointCloud, queue_size=2)
            self.map_count += 1
        '''

        self.measure_count = 0

    def start(self):

        rospy.init_node("mapper", anonymous=False)

        self.map_timer.start()
        #rospy.loginfo(f"mapper no. {self.map_count} started")
        
        rospy.spin()

    def prune_map(self, data):

        self.prune_timer.cancel()
        if self.locked == True or type(self.robot_particles) is type(None):
            self.prune_timer = threading.Timer(.001, self.prune_map, [data])
            self.prune_timer.start()
            return False
        else:
            self.locked = True

        measured_particles = self.decloud(data)
        if len(measured_particles) == 0:
            self.locked = False
            return True
    
        #new_particles = np.empty(np.shape(measured_particles))
        re_weights = set()

        for measured_i in range(np.shape(measured_particles)[0]):
            
            # add new particle measured distance from a random reference particle
            ref_index = np.random.randint(len(self.robot_particles))
            ref_particle = self.robot_particles[ref_index,:]

            #new_particles[measured_i, :] = self.transform_one(measured_particles[measured_i,:], ref_particle)
            #new_particles[measured_i, self.WEIGHT] = self.get_weight()

            ## downweight particles on path from reference to new particle
            # compute stats of new particles
            magnitude = np.sqrt(np.sum(measured_particles[measured_i,self.X:self.Y+1]**2))
            magnitude_scaling = .9
            measured_angle = 180/np.pi*np.arctan2(measured_particles[measured_i,self.Y],measured_particles[measured_i,self.X])
            angle = (measured_angle + ref_particle[self.ANGLE] + 180) % 360 - 180
            delta_angle = 30 #degrees

            # compute stats of existing particles and select ones on path
            particle_inds_to_examine = random.sample(range(len(self.particles)), 1000)
            reweighted_count = 0
            delta_Y = self.particles[particle_inds_to_examine,self.Y]-ref_particle[self.Y]
            delta_X = self.particles[particle_inds_to_examine,self.X]-ref_particle[self.X]
            robot_to_map_vecs_YX = np.hstack([np.reshape(delta_Y,[-1,1]), np.reshape(delta_X,[-1,1])])
            base_angles = 180/np.pi*np.arctan2(robot_to_map_vecs_YX[:,0],robot_to_map_vecs_YX[:,1])
            
            for map_particles_i in range(len(particle_inds_to_examine)):
                test_magnitude = np.sqrt(np.sum(robot_to_map_vecs_YX[map_particles_i,:]**2))
                test_angle = (base_angles[map_particles_i] - angle + 540) % 360 - 180

                #if test_magnitude < 1 and map_particles_i % 10 == 0:
                #    rospy.loginfo(f'test: {test_magnitude}, mag: {magnitude}, angle: {test_angle}')
                if test_magnitude < magnitude_scaling*magnitude and -delta_angle < test_angle and test_angle < delta_angle:
                    re_weights.add(particle_inds_to_examine[map_particles_i])
                    reweighted_count += 1
            
        # actually downweight the selected particles
        indecies = list(re_weights)
        self.particles[indecies,self.WEIGHT] = self.particles[indecies,self.WEIGHT]/2
        #rospy.loginfo(reweighted_count)

        #self.particles = np.vstack([self.particles, new_particles])
        #self.particle_data = np.hstack([self.particle_data, [{}]*len(new_particles)])

        self.publish()
        self.locked = False
        if self.measure_count % 3 == 0:
            self.measure_count = 1
            resampled = self.resample()
            return resampled
        else:
            self.measure_count += 1
            #rospy.loginfo("done; thread count: " + str(threading.active_count()))
            return True

    def grow_map(self, data):
        
        self.grow_timer.cancel()
        if self.locked == True or type(self.robot_particles) == type(None):
            self.grow_timer = threading.Timer(.001, self.grow_map, [data])
            self.grow_timer.start()
            return False
        else:
            self.locked = True

        new_particles = self.decloud(data)
        new_particles[:, self.WEIGHT] = new_particles[:, self.WEIGHT]*self.get_weight()
        self.particles = np.vstack([self.particles, new_particles])
        self.particle_data = np.array([{}]*len(self.particles))

        self.publish()
        self.locked = False
        resampled = self.resample()
        
        return resampled

    def get_weight(self):
        #return 1/(1 + 1e-2*(time.time()-self.start_time))
        return time.time()


    def reweight(self):
        return self.reweight_linear()

    def reweight_linear(self):
        new_weight = np.empty(np.shape(self.particles)[0])

        # set zeros to baseline weight
        baseline_weight = 1/2
        zeros = self.particles[:,self.WEIGHT] == 0
        new_weight[zeros] = baseline_weight

        # otherwise calculate weight as linear function of age
        slope = 1/60

        X = time.time() - self.particles[:,self.WEIGHT]
        new_weight[np.logical_not(zeros)] = slope*(X[np.logical_not(zeros)] + 1)

        return new_weight/np.max(new_weight)

    def reweight_sigmoid(self):
        #possible reweight function to call from "reweight"
        #this uses a sigmoid function

        new_weight = np.empty(np.shape(self.particles)[0])

        # set zeros to baseline weight
        baseline_weight = .1
        zeros = self.particles[:,self.WEIGHT] == 0
        new_weight[zeros] = baseline_weight

        # otherwise calculate weight as sigmoid of timestamp age
        x_at_90_percent = 30
        percent_at_intercept = .1

        X = time.time() - self.particles[:,self.WEIGHT]
        new_weight[np.logical_not(zeros)] = sigmoid((np.log(.1)/x_at_90_percent)*X[np.logical_not(zeros)]) + (percent_at_intercept - .5)

        return new_weight

    def update_robot(self, data):
        self.robot_particles = self.decloud(data)

    def publish(self):
        # restart timer
        self.map_timer.cancel()
        self.map_timer = threading.Timer(self.options["publish_interval"], self.publish)
        self.map_timer.start()
        
        self.map_pub.publish(self.make_cloud(self.particles))


if __name__ == "__main__":
    
    options = pf.FilterOptions(
        {
            "publish_interval": 1e6,
            "resample_interval": 1e6,
            "reset_weight_on_resample": False,
            "resample_noise_count": 10
        })
    map = Mapper(options)
    map.start()

