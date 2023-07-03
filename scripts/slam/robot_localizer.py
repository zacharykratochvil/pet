#!/usr/bin/python
"""ROS node for localization as part of a broader SLAM package

This module integrates sensor data and motor commmands from the robotic pet
into a particle filter for robot localization.

Typical usage example:

Call as part of a launch file that also starts several ROS node dependencies
such as the mapper and sensor nodes.

Copyright Zachary Kratochvil, 2022. All rights reserved.

"""
import common.particle_filter as pf
import rospy
import cv2
import time
import copy
import random
import heapq
import numpy as np
import threading
import sklearn.neighbors as skln
from integrator import Integrator
from common.accumulator import Accumulator, UltraSonicAccumulator
from common import vision_features as vf
from scipy.special import expit as sigmoid
import sklearn.cluster as sklc

#from mapper import Mapper

from dynamic_reconfigure.server import Server
from pet.cfg import RobotLocalizerConfig
from freenove_ros.msg import TwistDuration, SensorDistance
from sensor_msgs.msg import Imu, PointCloud
from geometry_msgs.msg import Point32, Twist
from std_msgs.msg import Float32
from pet.msg import Localization, LocationCandidate

class RobotLocalizer(pf.ParticleFilter):
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, options=pf.FilterOptions()):
        """Connects to the next available port.

        Args:
            options: 
                move_interval: 
                publish_interval: 

        Returns:
            The new robot localization filter.

        """
        options.options["initial_linear_dist"] = pf.UniformDistribution2D((-.1,.1),(-.1,.1))
        options.options["brick_threshold"] = 1e6
        super().__init__(options)

        self.inner_options = pf.FilterOptions({"initial_linear_dist":pf.UniformDistribution2D((-10,10),(-10,10)),
                                                  "resample_interval":1e6,
                                                  "use_timers": False,
                                                  "num_particles":30,
                                                  "resample_noise_count":np.nan,
                                                  "brick_threshold": 1e6
                                                  })
        #self.measure_timers = []
        for i in range(len(self.particles.ref)):
            self.particles.data[i]["map"] = pf.ParticleFilter(self.inner_options)
            '''
            def temp(self):
                indecies = list(range(np.shape(self.particles.ref)[0]))
                inds_to_keep = random.sample(indecies, k=self.particles.data[i]["map"].options["num_particles"])
                self.particles.ref = self.particles.ref[inds_to_keep,:]
                self.particles.data = self.particles.data[inds_to_keep]
                self.particles.regenerate_hash()
            self.particles.data[i]["map"].resample = temp
            '''
            #self.measure_timers.append(threading.Timer(1e6,None))
            #rospy.loginfo(self.particles.data[i]["map"].particles[:,0:2])
        

        self.integrator = Integrator()
        self.ultra_accumulator = UltraSonicAccumulator()
        #self.vision_accumulator = Accumulator()
        self.latest_map = None
        self.config = None

        self.move_timer = threading.Timer(-1, self.move)
        self.weight_timer = threading.Timer(-1, self.weight)
        self.measure_timer = threading.Timer(-1, self.measure)
        self.particle_timer = threading.Timer(-1, self.publish_particles)
        self.location_timer = threading.Timer(1, self.publish_location)

        self.drive_sub = rospy.Subscriber("drive_twist", TwistDuration, self.integrator.on_twist, queue_size=10)
        self.witmotion_sub = rospy.Subscriber("imu", Imu, self.integrator.on_odo, queue_size=1)
        self.ultra_sub = rospy.Subscriber("ultrasonic_distance", SensorDistance, self.ultra_accumulator.on_ultra, queue_size=1)
        #self.optical_sub = rospy.Subscriber("optical_velocity", Twist, self.integrator.on_vision, queue_size=1)
        self.map_sub = rospy.Subscriber("map", PointCloud, self.map, queue_size=1)

        self.measurement_pub = rospy.Publisher("measured_particles", PointCloud, queue_size=1)
        self.particle_pub = rospy.Publisher("robot_particles", PointCloud, queue_size=1)
        self.local_particle_pub = rospy.Publisher("local_map_particles", PointCloud, queue_size=1)
        self.location_pub = rospy.Publisher("robot_location", Localization, queue_size=1)

    def start(self):
        """Connects to the next available port.

        Args:
            minimum: A port value greater or equal to 1024.

        Returns:
            The new minimum port.

        """
        rospy.init_node("robot_localizer", anonymous=False)

        self.cfg_srv = Server(RobotLocalizerConfig, self.cfg_callback)


        #Timers are started by dynamic reconfigure now
        '''
        self.move_timer.start()
        self.weight_timer.start()
        self.measure_timer.start()
        self.particle_timer.start()
        '''
        self.location_timer.start()

        self.publish_particles()
        rospy.loginfo(self.particles.data[0]["map"].particles.ref[:,0:2])
        rospy.loginfo(self.particles.data[1]["map"].particles.ref[:,0:2])

        rospy.spin()

    def cfg_callback(self, config, level):
        
        # update this particle filter's resample properties
        # this has been disabled because it interacts with local map particle updates,
        # and because resize is not supported
        #if config["base_localizer_particles"] != self.options["num_particles"]:
        #    self.resize(config["base_localizer_particles"])
        self.options["resample_noise_count"] = config["localizer_noise_count"]

        # update local map particle filters' resample properties
        # disabled because resize is not supported
        #if config["local_map_particles"] != self.particles.data[0]["map"].options["num_particles"]:
        #    for i in range(self.options["num_particles"]):
        #        self.particles.data[i]["map"].resize(config["local_map_particles"])
        if config["local_map_noise_count"] != self.particles.data[0]["map"].options["resample_noise_count"]:
            for i in range(self.options["num_particles"]):
                self.particles.data[i]["map"].options["resample_noise_count"] = config["local_map_noise_count"]

        # update timers
        if self.options["move_interval"] != config["move_interval"]:
            self.options["move_interval"] = config["move_interval"]
            self.move_timer.cancel()
            self.move_timer = threading.Timer(self.options["move_interval"], self.move)
            self.move_timer.start()

        if self.options["measure_interval"] != config["measure_interval"]:
            self.options["measure_interval"] = config["measure_interval"]
            self.measure_timer.cancel()
            self.measure_timer = threading.Timer(self.options["measure_interval"], self.measure)
            self.measure_timer.start()

        if self.options["weight_interval"] != config["weight_interval"]:
            self.options["weight_interval"] = config["weight_interval"]
            self.weight_timer.cancel()
            self.weight_timer = threading.Timer(self.options["weight_interval"], self.weight)
            self.weight_timer.start()

        if self.options["publish_interval"] != config["publish_interval"]:
            self.options["publish_interval"] = config["publish_interval"]
            self.particle_timer.cancel()
            self.particle_timer = threading.Timer(self.options["publish_interval"], self.publish_particles)
            self.particle_timer.start()
        
        # update measurement properties
        self.options["local_map_update_subset_factor"] = config["local_map_update_subset_factor"]

        self.config = config
        return self.config

    def resample(self, ignore_lock=False):

        @pf.ParticleFilter.locking(calling_fn="resample", timer_name="resample_timer", long_timeout=self.options["resample_interval"], ignore_lock=ignore_lock)
        def inner_resample(self, *args, **kwargs):
            super().resample(ignore_lock=True)

            for i in range(len(self.particles.ref)):#-self.options["resample_noise_count"], len(self.particles)):
                #self.particles.data[i]["map"].close()
                #self.particles.data[i]["map"] = pf.ParticleFilter(self.inner_options)
                if "map" in self.particles.data[i].keys():
                    #options = pf.FilterOptions({"num_particles":self.options["num_particles"], "brick_threshold":self.options["brick_threshold"]})
                    self.particles.data[i]["map"].particles = pf.Particles(self.inner_options)
                else:
                    self.particles.data[i]["map"] = pf.ParticleFilter(self.inner_options)
                    '''
                    def temp(self):
                        indecies = list(range(np.shape(self.particles.ref)[0]))
                        inds_to_keep = random.sample(indecies, k=self.particles.data[i]["map"].options["num_particles"])
                        self.particles.ref = self.particles.ref[inds_to_keep,:]
                        self.particles.data = self.particles.data[inds_to_keep]
                        self.particles.regenerate_hash()
                    self.particles.data[i]["map"].resample = temp
                    '''

            for i in range(len(self.particles.ref)):
                #rospy.loginfo(self.particles.data[i]["map"].particles[0])
                if i > 0:
                    pass#rospy.logerr(f'resample particles: {self.particles.data[i]["map"].particles is self.particles.data[i-1]["map"].particles}')

            for i in range(len(self.particles.ref)):
                #rospy.loginfo(self.particles.data[i]["map"].particles[0])
                if i > 0:
                    pass#rospy.logerr(f'resample dict: {self.particles.data[i] is self.particles.data[i-1]}')

        start = time.time()
        inner_resample(self)
        #rospy.logerr(f"resample: {time.time()-start}")

    def map(self, data):
        self.latest_map = pf.ParticleFilter.decloud(data)
        
    def move(self):
        """Connects to the next available port.

        Args:
            minimum: A port value greater or equal to 1024.

        Returns:
            The new minimum port.

        """

        @pf.ParticleFilter.locking(calling_fn="move", timer_name="move_timer", long_timeout=self.options["move_interval"])
        def inner_move(self, *args, **kwargs):

            delta_pos, var = self.integrator.step()
            rospy.loginfo(f"pos: {delta_pos}; var: {var}")

            i = 0
            variance_offset = 0
            was_valid = np.zeros([np.shape(self.particles.ref)[0], 2])
            while np.sum(was_valid) < np.shape(self.particles.ref)[0]*2:

                # produce a list of random errors to apply to particles
                i += 1
                variance_multiplier = .7 #.5
                variance_offset += .1*i #.05
                _args_list = (delta_pos["linear_pos"][0], delta_pos["linear_pos"][1], variance_multiplier*var["linear_pos"][0] + variance_offset, variance_multiplier*var["linear_pos"][1] + variance_offset)
                lin_error_dist = pf.GaussianDistribution2D(*_args_list)
                lin_errors = lin_error_dist.draw(self.options["num_particles"])
                not_was_valid = -1*(was_valid-1)
                lin_errors = not_was_valid*lin_errors
                
                #lin_errors = np.hstack([lin_errors, np.zeros([self.options["num_particles"],1])])

                for particle_i in range(np.shape(self.particles.ref)[0]):
                    interpolation = [None,None]
                    distance = np.sqrt(np.sum(self.particles.ref[particle_i,0:2]**2 + (self.particles.ref[particle_i,0:2]+lin_errors[particle_i,0:2])**2))
                    for xy in [0,1]:
                        interpolation[xy] = np.asarray(np.linspace(self.particles.ref[particle_i,xy], self.particles.ref[particle_i,xy]+lin_errors[particle_i,xy], int(np.ceil(distance/.2))))
                    
                    was_valid[particle_i,:] = int(0 == np.sum([self.particles.is_brick([x,y]) for x,y in zip(interpolation[0],interpolation[1])]))
                
            ang_errors = np.random.normal(delta_pos["angular_pos"][2],var["angular_pos"][2],self.options["num_particles"])

            # apply deltas to particles
            weights = np.zeros([self.options["num_particles"],1])
            particle_deltas = np.hstack([lin_errors[:,0:2], np.reshape(ang_errors,[-1,1]), weights])
            self.particles.ref += particle_deltas
            self.particles.ref[:,self.ANGLE] = ((self.particles.ref[:,self.ANGLE] + 180) % 360) - 180
            self.particles.regenerate_hash()
            self.particles.regenerate_weight()

            self.publish_particles()
        
            return True

        start = time.time()
        inner_move(self)
        #rospy.logerr(f"move: {time.time()-start}")

    def measure(self):
        """Handles measurement inputs.

        Args:
            

        Returns:
            

        """

        '''
        self.measure_timer.cancel()
        self.measure_timer = threading.Timer(self.options["measure_interval"], self.measure)
        self.measure_timer.start()
        '''

        @pf.ParticleFilter.locking("measure", timer_name="measure_timer", long_timeout=self.options["measure_interval"])
        def inner_measure(self, *args, **kwargs):

            # generate short list of points, convert cm to m and scatter in perpendicular direction
            distances = np.array(self.ultra_accumulator.get_data())/100
            if len(distances) == 0:
                return True

            self.integrator.update_distances(distances)

            selected_distances = np.random.choice(distances,max(len(distances),20))
            scatter = [np.random.normal(0,np.abs(np.tan(15/180*np.pi)*dist)) for dist in selected_distances]
            xy = np.hstack([np.reshape(scatter,[-1,1]),np.reshape(selected_distances,[-1,1])])
            measured_particles = np.hstack([xy, np.zeros([np.shape(xy)[0], 2])])

            # publish points
            pc = self.make_cloud(measured_particles)
            self.measurement_pub.publish(pc)

            # update a random subset of local maps with a random subset of measured particles
            
            for i in random.sample(range(len(self.particles.ref)), int(np.ceil(len(self.particles.ref)/self.options["local_map_update_subset_factor"]))):
                
                num_measureds_to_sample = len(measured_particles) #int(np.ceil(len(measured_particles)/3))
                added_particles = np.empty([num_measureds_to_sample, 4])   
                added_i = 0

                for measured_i in random.sample(range(len(measured_particles)), num_measureds_to_sample):
                    added_particles[added_i,:] = (self.transform_one(measured_particles[measured_i,:], self.particles.ref[i]))
                    added_i += 1

                #rospy.logerr(added_particles[0])

                all_particles = np.vstack([copy.deepcopy(self.particles.data[i]["map"].particles.ref), added_particles])
                self.particles.data[i]["map"].particles.ref = all_particles
                self.particles.data[i]["map"].particles.data = np.array([{} for i in range(np.shape(self.particles.data[i]["map"].particles.ref)[0])])
                self.particles.data[i]["map"].particles.regenerate_weight()
                self.particles.data[i]["map"].particles.regenerate_hash()
                #rospy.logerr(len(self.particles.data[i]["map"].particles))
               
                '''
                @pf.ParticleFilter.locking(f"particles.data[{i}]['map'].resample", timer_name=f"measure_timers[{i}]", short_timeout=.01, long_timeout=.01)
                def measure_resample(self, *args, **kwargs):
                    return self.particles.data[i]["map"].resample()
                measure_resample(self)
                '''
                
                #rospy.logerr(f"particle {added_particles[0,:]}")
                #self.particles.data[i]["map"].resample()
                #if ("map" in self.particles.data[i].keys()) == False:
                #    self.particles.data[i]["map"] = pf.ParticleFilter(self.inner_options)

            for i in range(np.shape(self.particles.ref)[0]):
                #rospy.loginfo(self.particles.data[i]["map"].particles[0])
                if i > 0:
                    pass#rospy.logerr(f'measure: {self.particles.data[i]["map"].particles is self.particles.data[i-1]["map"].particles}')
            
            return True

        start = time.time()
        inner_measure(self)
        #rospy.logerr(f"measure: {time.time()-start}")


    def weight(self):
        """Handles measurement inputs.

        Args:
            

        Returns:
            

        """
        if type(self.latest_map) == type(None):
            self.publish_particles()
            self.weight_timer.cancel()
            self.weight_timer = threading.Timer(.1, self.weight)
            self.weight_timer.start()
            rospy.logerr("weight aborted")
            return False
        
        @pf.ParticleFilter.locking("weight", timer_name="weight_timer", long_timeout=self.options["weight_interval"])
        def inner_weight(self, *args, **kwargs):


            # compare to map, score
            #options = pf.FilterOptions({"num_particles":self.options["num_particles"], "brick_threshold":self.options["brick_threshold"]})
            self.latest_map_particles = pf.Particles(self.inner_options, ref=self.latest_map)

            '''
            num_neighbors = 10
            nn_tree = skln.NearestNeighbors(n_neighbors = num_neighbors, algorithm = "kd_tree",
                                                leaf_size = 10, n_jobs = 1)
            
            if np.any(np.isnan(self.latest_map[:,0:2])):
                rospy.logerr("latest map is nan")
            nn_tree.fit(self.latest_map[:,0:2])

            #for i in range(self.options["num_particles"]):
            #    rospy.logerr(self.particles.data[i]["map"].particles[0])
            '''
            
            #avg_distances = []
            norm_neighbors = np.empty(np.shape(self.particles.ref)[0])
            for i in range(np.shape(self.particles.ref)[0]):
                local_particle_coords = self.particles.data[i]["map"].particles.ref[:,0:2]
                num_radius_neighbors = np.zeros(1)
                rospy.logerr(np.shape(local_particle_coords)[0])
                for ii in range(np.shape(local_particle_coords)[0]):
                    num_radius_neighbors += len(self.latest_map_particles.get_inds(local_particle_coords[ii,:]))
                    num_radius_neighbors += self.latest_map_particles.options["brick_threshold"]*self.latest_map_particles.is_brick(local_particle_coords[ii,:])

                #rospy.loginfo(self.particles.data[i]["map"].particles[:,0:2])
                #distances = neighbors.flatten()
                #distances.sort()
                #distance = np.mean(distances)#[:int(len(distances)*.2)])
                
                #normalize by number of particles searched
                rospy.logerr("after the for loop")
                norm_neighbors_one = (num_radius_neighbors)/np.shape(local_particle_coords)[0]
                #don't add extra weight for extremely dense regions
                max_density = 15
                norm_neighbors_one = np.min([max_density, norm_neighbors_one])
                norm_neighbors[i] = norm_neighbors_one
                #avg_distances.append(distance)
                rospy.logerr(f"reweight distance: {norm_neighbors[i]}")

            min_dist = np.min(norm_neighbors)
            max_dist = np.max(norm_neighbors)
            if min_dist == max_dist:
                return True

            local_particles = []
            for i in range(len(norm_neighbors)):

                map_weight = (norm_neighbors[i] - min_dist)/(max_dist - min_dist)
                odo_z_orientation = self.integrator.get_orientation()[2]
                particle_z_orientation = self.particles.ref[i,self.ANGLE]
                odo_weight = 1-np.abs((((odo_z_orientation - particle_z_orientation) + 540) % 360) - 180)/180
                weight = map_weight*odo_weight
                #rospy.logerr(f"magnetic: {odo_z_orientation}; filter: {particle_z_orientation}")

                if np.isinf(weight):
                    weight = 0
                elif weight < 0:
                    weight = 0
                self.particles.ref[i,self.WEIGHT] = weight

                particle_copy = self.particles.data[i]["map"].particles.ref
                new_local_particles = np.empty([np.shape(particle_copy)[0], 4])
                new_local_particles[:,0:2] = particle_copy[:,0:2]
                new_local_particles[:,self.ANGLE] = 0
                new_local_particles[:,self.WEIGHT] = weight
                local_particles.append(new_local_particles)
                #self.particles.data[i]["map"].particles = pf.Particles(ref=new_local_particles)

            self.particles.regenerate_weight()
            local_particles = np.vstack(local_particles)
            pc = self.make_cloud(local_particles)
            self.local_particle_pub.publish(pc)

            #for i in range(len(self.particles)):
                #rospy.loginfo(self.particles.data[i]["map"].particles[0])
             #   if i > 0:
             #       pass#rospy.logerr(f'weight: {self.particles.data[i]["map"].particles is self.particles.data[i-1]["map"].particles}')

            self.publish_particles()
            self.resample(ignore_lock=True)

            return True

        start = time.time()
        inner_weight(self)
        #rospy.logerr(f"weighting: {time.time()-start}")

        return True

    def publish_particles(self):
        self.particle_timer.cancel()
        self.particle_timer = threading.Timer(self.options["publish_interval"], self.publish_particles)
        self.particle_timer.start()

        pc = self.make_cloud(self.particles.ref)
        self.particle_pub.publish(pc)

    def publish_location(self):
        self.location_timer.cancel()
        self.location_timer = threading.Timer(1, self.publish_location)
        self.location_timer.start()

        start = time.time()

        # find largest cluster in robot map
        candidates = []

        '''
        grid_spacing = np.arange(-10,10.01,.5)
        self.seeds = []
        for x in grid_spacing:
            for y in grid_spacing: 
                self.seeds.append((x,y))
        
        tuples = []
        bandwidth = 1.5
        discretized_particles = np.round(self.particles[:,0:2]*bandwidth)/bandwidth
        for i in range(np.shape(self.particles)[0]):
            tuples.append(tuple(discretized_particles[i,:]))
        seeds = np.vstack(list(set(tuples)))

        rospy.logerr(f"partial clustering 1: {time.time()-start}")
        '''

        #try:
        #ms = sklc.MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=1, max_iter=20) #seeds=seeds
        #ms.fit(self.particles[:,:2])
        
        cluster_center = vf.kernel_cluster_extreme(self.particles)
        
        cluster_particle_inds = self.particles.get_inds(cluster_center[0,:])
        cluster_particles = self.particles.ref[cluster_particle_inds,:]

        candidate = LocationCandidate()
        candidate.position.x = cluster_center[0,0]
        candidate.position.y = cluster_center[0,1]
        candidate.position.z = 0
        candidate.orientation.x = 0
        candidate.orientation.y = 0
        candidate.orientation.z = vf.angle_mean(cluster_particles[:,self.ANGLE])
        candidate.confidence = 1
        candidate.spread = np.mean(np.var(cluster_particles[:,0:2], axis=0))
        candidates.append(candidate)

        '''
        values, counts = np.unique(clustered_points, return_counts=True, axis=0)
        sorted_inds = np.argsort(counts) #ascending

        #rospy.logerr(f"partial clustering 1: {time.time()-start}")
        
        for i in np.flip(sorted_inds):
            is_this_cluster = np.all(clustered_points == np.tile(values[i,:],(np.shape(clustered_points)[0], 1)), axis=1)#ms.labels_ == values[i]            
            cluster_orientations = self.particles.ref[is_this_cluster,2]
            confidence = np.sum(is_this_cluster)/np.shape(self.particles.ref)[0]
            spread = np.mean([np.var(self.particles.ref[is_this_cluster,0]), np.var(self.particles.ref[is_this_cluster,1])])
    
            candidate = LocationCandidate()
            candidate.position.x = values[i,0]
            candidate.position.y = values[i,1]
            candidate.position.z = 0
            candidate.orientation.x = 0
            candidate.orientation.y = 0
            candidate.orientation.z = vf.angle_mean(cluster_orientations)
            candidate.confidence = confidence
            candidate.spread = spread
            candidates.append(candidate)
        #except ValueError as e:
        #    rospy.logerr(f"shape: {np.shape(self.particles)}")
        #    rospy.logerr(f"nan/inf: {np.any(np.isnan(self.particles))}")
        '''

        msg = Localization()
        msg.candidates = candidates
        self.location_pub.publish(msg)

        #rospy.logerr(f"clustering 2: {time.time()-start}")

if __name__ == "__main__":
    options = {
            "resample_interval": 1e6,
            
            #these are set in launch file through daynamic reconfigure but must still be initialized
            "move_interval": np.nan, 
            "measure_interval": np.nan,
            "weight_interval": np.nan,
            "publish_interval": np.nan,
            "resample_noise_count": np.nan,

            "num_particles": 15,
            #"local_map_update_subset_factor": 10
        }
    options = pf.FilterOptions(options)
    robot_filter = RobotLocalizer(options)
    robot_filter.start()