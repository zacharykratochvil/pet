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
import numpy as np
import threading
import sklearn.neighbors as skln
from common.accumulator import Accumulator, UltraSonicAccumulator
from scipy.special import expit as sigmoid

#from mapper import Mapper

from freenove_ros.msg import TwistDuration, SensorDistance
from sensor_msgs.msg import Imu, PointCloud
from geometry_msgs.msg import Point32, Twist
from std_msgs.msg import Float32

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
        super().__init__(options)

        self.inner_options = pf.FilterOptions({"initial_linear_dist":pf.UniformDistribution2D((-10,10),(-10,10)),
                                                  "resample_interval":1e6,
                                                  "use_timers": False,
                                                  "num_particles":50,
                                                  "resample_noise_count":0
                                                  })
        for i in range(len(self.particles)):
            self.particle_data[i]["map"] = pf.ParticleFilter(self.inner_options)
            #rospy.loginfo(self.particle_data[i]["map"].particles[:,0:2])
        
        self.integrator = Integrator()
        self.ultra_accumulator = UltraSonicAccumulator()
        #self.vision_accumulator = Accumulator()
        self.latest_map = None

        self.move_timer = threading.Timer(self.options["move_interval"], self.move)
        self.weight_timer = threading.Timer(self.options["weight_interval"], self.weight)
        self.measure_timer = threading.Timer(self.options["measure_interval"], self.measure)
        self.publish_timer = threading.Timer(self.options["publish_interval"], self.publish_particles)

        self.drive_sub = rospy.Subscriber("drive_twist", TwistDuration, self.integrator.on_twist, queue_size=10)
        self.witmotion_sub = rospy.Subscriber("imu", Imu, self.integrator.on_odo, queue_size=1000)
        self.ultra_sub = rospy.Subscriber("ultrasonic_distance", SensorDistance, self.ultra_accumulator.on_ultra, queue_size=1)
        self.optical_sub = rospy.Subscriber("optical_velocity", Twist, self.integrator.on_vision, queue_size=1)
        self.map_sub = rospy.Subscriber("map", PointCloud, self.map, queue_size=1)

        self.measurement_pub = rospy.Publisher("measured_particles", PointCloud, queue_size=1)
        self.particle_pub = rospy.Publisher("robot_particles", PointCloud, queue_size=1)
        self.local_particle_pub = rospy.Publisher("local_map_particles", PointCloud, queue_size=1)

    def start(self):
        """Connects to the next available port.

        Args:
            minimum: A port value greater or equal to 1024.

        Returns:
            The new minimum port.

        """
        rospy.init_node("robot_localizer", anonymous=False)
        
        self.move_timer.start()
        self.weight_timer.start()
        self.measure_timer.start()
        self.publish_timer.start()
        self.publish_particles()
        rospy.loginfo(self.particle_data[0]["map"].particles[:,0:2])
        rospy.loginfo(self.particle_data[1]["map"].particles[:,0:2])

        rospy.spin()

    def resample(self):
        if super().resample():
            for i in range(len(self.particles)-self.options["resample_noise_count"], len(self.particles)):
                self.particle_data[i]["map"] = pf.ParticleFilter(self.inner_options)

    def map(self, data):
        self.latest_map = self.decloud(data)
        
    def move(self):
        """Connects to the next available port.

        Args:
            minimum: A port value greater or equal to 1024.

        Returns:
            The new minimum port.

        """
        # restart timer
        self.move_timer.cancel()

        # only proceed if can obtain lock
        if self.locked == True:
            self.move_timer = threading.Timer(.001, self.move)
            self.move_timer.start()
            return False
        else:
            self.locked = True
            self.move_timer = threading.Timer(self.options["move_interval"], self.move)
            self.move_timer.start()


        delta_pos, var = self.integrator.step()
        #rospy.loginfo(f"pos: {delta_pos}; var: {var}")

        # produce a list of random errors to apply to particles
        variance_multiplier = 1.5
        variance_offset = .05
        _args_list = (delta_pos["linear_pos"][0], delta_pos["linear_pos"][1], variance_multiplier*var["linear_pos"][0] + variance_offset, variance_multiplier*var["linear_pos"][1] + variance_offset)
        lin_error_dist = pf.GaussianDistribution2D(*_args_list)
        lin_errors = lin_error_dist.draw(self.options["num_particles"])
        lin_errors = np.hstack([lin_errors, np.zeros([self.options["num_particles"],1])])

        ang_errors = np.random.normal(delta_pos["angular_pos"][2],var["angular_pos"][2],self.options["num_particles"])

        # apply deltas to particles
        weights = np.zeros([self.options["num_particles"],1])
        particle_deltas = np.hstack([lin_errors[:,0:2], np.reshape(ang_errors,[-1,1]), weights])
        self.particles += particle_deltas
        self.particles[:,self.ANGLE] = ((self.particles[:,self.ANGLE] + 180) % 360) - 180

        self.publish_particles()
        self.locked = False
        return True


    def measure(self):
        """Handles measurement inputs.

        Args:
            

        Returns:
            

        """

        # restart timer
        self.measure_timer.cancel()        

        # only proceed if can obtain lock
        if self.locked == True:
            self.measure_timer = threading.Timer(.001, self.measure)
            self.measure_timer.start()
            return False
        else:
            self.locked = True
            self.measure_timer = threading.Timer(self.options["measure_interval"], self.measure)
            self.measure_timer.start()

        # generate short list of points, convert cm to m and scatter in perpendicular direction
        distances = np.array(self.ultra_accumulator.get_data())/100
        if len(distances) == 0:
            self.locked = False
            return True

        self.integrator.update_distances(distances)

        selected_distances = np.random.choice(distances,max(len(distances),10))
        scatter = [np.random.normal(0,np.abs(np.tan(15/180*np.pi)*dist)) for dist in selected_distances]
        xy = np.hstack([np.reshape(scatter,[-1,1]),np.reshape(selected_distances,[-1,1])])
        measured_particles = np.hstack([xy, np.zeros([np.shape(xy)[0], 2])])

        # publish points
        pc = self.make_cloud(measured_particles)
        self.measurement_pub.publish(pc)

        # update a random subset of local maps with a random subset of measured particles
        for i in np.random.randint(low=0, high=len(self.particles), size=int(np.ceil(len(self.particles)/5))):
            
            num_measureds_to_sample = len(measured_particles) #int(np.ceil(len(measured_particles)/3))
            added_particles = np.empty([num_measureds_to_sample, 4])   
            added_i = 0

            for measured_i in random.sample(range(len(measured_particles)), num_measureds_to_sample):
                added_particles[added_i,:] = (self.transform_one(measured_particles[measured_i,:], self.particles[i]))
                added_i += 1

            all_particles = np.vstack([self.particle_data[i]["map"].particles, added_particles])
            self.particle_data[i]["map"].particles = all_particles
            self.particle_data[i]["map"].particle_data = np.array([{} for i in range(len(self.particle_data[i]["map"].particles))])
            self.particle_data[i]["map"].resample()
        
        self.locked = False
        return True


    def weight(self):
        """Handles measurement inputs.

        Args:
            

        Returns:
            

        """
        # restart timer
        self.weight_timer.cancel()

        # only proceed if can obtain lock
        if type(self.latest_map) == type(None):
            self.publish_particles()
            return False
        elif self.locked == True:
            self.weight_timer = threading.Timer(.001, self.weight)
            self.weight_timer.start()
            return False
        else:
            self.locked = True
            self.weight_timer = threading.Timer(self.options["weight_interval"], self.weight)
            self.weight_timer.start()

        rospy.loginfo("attempting to weight")

        # compare to map, score
        num_neighbors = 20
        nn_tree = skln.NearestNeighbors(n_neighbors = num_neighbors, algorithm = "kd_tree",
                                            leaf_size = 10, n_jobs = 4)
        nn_tree.fit(self.latest_map[:,0:2])

        local_particles = []
        for i in range(self.options["num_particles"]):
            neighbors = nn_tree.kneighbors(self.particle_data[i]["map"].particles[:,0:2], n_neighbors=num_neighbors, return_distance = True)
            #rospy.loginfo(self.particle_data[i]["map"].particles[:,0:2])
            distances = neighbors[0].flatten()
            #distances.sort()
            distance = np.mean(distances)#[:int(len(distances)*.8)])
            weight = 1 - np.tanh(distance)
            self.particles[i,self.WEIGHT] = max(0, weight)

            new_local_particles = np.empty([len(self.particle_data[i]["map"].particles), 4])
            new_local_particles[:,0:2] = self.particle_data[i]["map"].particles[:,0:2]
            new_local_particles[:,self.ANGLE] = 0
            new_local_particles[:,self.WEIGHT] = weight
            local_particles.append(new_local_particles)

        local_particles = np.vstack(local_particles)
        pc = self.make_cloud(local_particles)
        self.local_particle_pub.publish(pc)

        self.publish_particles()
        self.locked = False

        self.resample()
        return True

    def publish_particles(self):
        self.publish_timer.cancel()
        self.publish_timer = threading.Timer(self.options["publish_interval"], self.publish_particles)
        self.publish_timer.start()

        pc = self.make_cloud(self.particles)
        self.particle_pub.publish(pc)


class Integrator:
    """Integrates motor commands and odometry signals.

    Longer class information...
    Longer class information...

    Attributes:
        latest_twist: dictionary compactly storing the most recent twist message received
        latest_odo: dictionary compactly storing the most recent odometry message received
        latest_stamp: dictionary with the timestamp of the most recent message received of any kind
        vel_integral_odo:
        pos_integral:
        pos_variance:
        VELOCITY_MAX:

    """

    def __init__(self):
        """Connects to the next available port.

        Args:
            minimum: A port value greater or equal to 1024.

        Returns:
            The new minimum port.

        Raises:
            ConnectionError: If no available port is found.
        """
        
        # constants
        self.VELOCITY_MAX = 2 #m/s

        # stores last measurements
        self.latest_twist = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.latest_vision = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.latest_odo = {"linear_acc":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.latest_corr_linear_acc = (0.0,0.0,0.0)
        self.latest_store_stamp = time.time()
        self.latest_access_stamp = time.time()
        self.latest_distances = None

        self.prev_vision = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.prev_odo = {"linear_acc":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.prev_corr_linear_acc = (0.0,0.0,0.0)
        self.prev_store_stamp = time.time()
        self.prev_access_stamp = time.time()

        # stores odometer estimated linear velocities
        self.vel_integral_odo = {"linear_vel":np.array([0.0,0.0,0.0])}

        # zero integrals
        self.pos_integral = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}
        self.last_access_pos_integral = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}
        self.pos_variance = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}

        # twist callback        
        self.timer = threading.Timer(0, self._clear_twist)

    def update_distances(self, distances):
        self.latest_distances = distances

    def _clear_twist(self):
        self.update_integral(time.time())
        self.latest_twist = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}

    #average with heavier weight to lower value
    def _smooth(self, array1, array2, weight_ratio=5):
        min_index = np.argmin([array1, array2], axis = 0)

        weights = np.ones([2,3])
        for i in range(len(min_index)):
            weights[min_index[i],i] = weight_ratio

        return np.average([array1, array2], axis = 0)

    def on_twist(self, msg):
        """Processes and integrates Twist drive request messages.

        Args:
            msg: the TwistDuration message received by the subscriber
            
        """
        # update timers
        self.timer.cancel()
        #duration = msg.duration.secs + msg.duration.nsecs*1e-9
        self.timer = threading.Timer(msg.duration, self._clear_twist)
        self.timer.start()

        # update integrals
        self.update_integral(time.time())

        # update twists
        self.latest_twist["linear_vel"] = (
                msg.velocity.linear.x,
                msg.velocity.linear.y,
                msg.velocity.linear.z
            )
        self.latest_twist["angular_vel"] = (
                msg.velocity.angular.x,
                msg.velocity.angular.y,
                msg.velocity.angular.z
            )

    def on_vision(self, msg):
        if type(self.latest_distances) == type(None):
            return

        self.prev_vision = copy.deepcopy(self.latest_vision)

        x_vel = msg.linear.x - np.pi/180*self._calc_angular_vel()[2]*np.mean(self.latest_distances)
        self.latest_vision["linear_vel"] = [x_vel, msg.linear.y, msg.linear.z]
        for i in range(3):
            if np.squeeze(np.abs(self.latest_vision["linear_vel"][i])) > self.VELOCITY_MAX:
                self.latest_vision["linear_vel"][i] = np.sign(self.latest_vision["linear_vel"])*self.VELOCITY_MAX[i]
        
        self.latest_vision["linear_vel"] = self._smooth(self.prev_vision["linear_vel"], self.latest_vision["linear_vel"])

        self.update_integral(time.time())

    def on_odo(self, msg):
        """Processes and integrates Imu messages.

        Args:
            msg: the Imu message received by the subscriber to imu

        """

        # store previous and update latest
        self.prev_odo = copy.deepcopy(self.latest_odo)
        self.latest_odo["linear_acc"] = (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            )
        self.latest_odo["angular_vel"] = (
                msg.angular_velocity.x*180/np.pi,
                msg.angular_velocity.y*180/np.pi,
                -msg.angular_velocity.z*180/np.pi
            )

        stamp = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9

        # calculate correction of linear acceleration for gravity
        delta_time = stamp - self.latest_store_stamp
        _angular_integral = self.pos_integral["angular_pos"] + self._calc_angular_vel()*delta_time
        _pos_integral = {
                "angular_pos": _angular_integral
            }

        g = -9.81 # m/s**2
        if np.sum(np.abs(self.latest_odo["angular_vel"])) == 0:
            x_percent_g = max(-1.0,min(1.0,self.latest_odo["linear_acc"][0]/g))
            y_percent_g = max(-1.0,min(1.0,self.latest_odo["linear_acc"][1]/g))
            x = np.arcsin(y_percent_g)
            y = -np.arcsin(x_percent_g)

            self.pos_integral["angular_pos"] = (x, y, self.pos_integral["angular_pos"][2])
            _pos_integral["angular_pos"] = self.pos_integral["angular_pos"]

        g_x = -g*np.sin(_pos_integral["angular_pos"][1]/180*np.pi)
        g_y = g*np.sin(_pos_integral["angular_pos"][0]/180*np.pi)
        g_z = -np.sqrt(max(0, g**2 - g_x**2 - g_y**2))
        g_correction = np.asarray((g_x, g_y, g_z))

        self.prev_corr_linear_acc = copy.deepcopy(self.latest_corr_linear_acc)
        if np.sum(np.abs(self.latest_odo["angular_vel"])) == 0:
            self.latest_corr_linear_acc = (0.0, 0.0, 0.0)
            self.vel_integral_odo = {"linear_vel":(0.0, 0.0, 0.0)}
        else:
            self.latest_corr_linear_acc = self.latest_odo["linear_acc"] - g_correction

        #self.latest_odo["linear_vel"] = self._smooth()

        # calculate position integrals
        self.update_integral(stamp)

    def _calc_angular_vel(self):
        return np.mean(np.vstack(
                    [
                        self.prev_odo["angular_vel"],
                        self.latest_odo["angular_vel"]
                        #self.latest_twist["angular_vel"]
                    ]
                ),0)

    def update_integral(self, stamp):
        """Performs most of the integration logic.

        Should be called anytime any of the integrator's inputs are updated.

        """

        # update time
        self.prev_store_stamp = self.latest_store_stamp
        self.latest_store_stamp = stamp #time.time()
        delta_time = self.latest_store_stamp - self.prev_store_stamp

        # update integrals
        decay_constant = .95
        prev_vel_integral_odo = copy.deepcopy(self.vel_integral_odo)
        self.vel_integral_odo["linear_vel"] += np.mean((
                self.prev_corr_linear_acc,
                self.latest_corr_linear_acc
            ),0)*delta_time*decay_constant
        #self.vel_integral_odo["linear_vel"] = np.mean((self.vel_integral_odo["linear_vel"],
        #        self.latest_twist["linear_vel"]),0)
        for i in range(3):
            _vel_pointer = self.vel_integral_odo["linear_vel"]
            if np.abs(_vel_pointer[i]) > self.VELOCITY_MAX:
                _vel_pointer[i] = np.sign(_vel_pointer[i])*self.VELOCITY_MAX

        
        #rospy.loginfo("prev_integral: " + str(prev_vel_integral_odo["linear_vel"]))
        #rospy.loginfo(self.vel_integral_odo["linear_vel"])
        #rospy.loginfo(self.latest_twist["linear_vel"])

        self.pos_integral["linear_pos"] += np.nanmean((
                #prev_vel_integral_odo["linear_vel"],
                #self.vel_integral_odo["linear_vel"],
                #self.latest_vision["linear_vel"],
                #self.latest_vision["linear_vel"],
                self.prev_vision["linear_vel"],
                self.latest_vision["linear_vel"]
                #self.latest_twist["linear_vel"]
                #self.latest_twist["linear_vel"]
            ), axis = 0)*delta_time

        #rospy.loginfo("integral: " + str(self.pos_integral["linear_pos"]))
        self.pos_integral["angular_pos"] += self._calc_angular_vel()*delta_time
        self.pos_integral["angular_pos"] = ((self.pos_integral["angular_pos"] + 540) % 360) - 180

        #rospy.logwarn(self.pos_integral["angular_pos"])

        #self.pos_variance["linear_pos"] += np.nanvar((
        #        prev_vel_integral_odo["linear_vel"],
        #        self.vel_integral_odo["linear_vel"],
        #        self.latest_vision["linear_vel"]
                #self.latest_twist["linear_vel"]
                #self.latest_twist["linear_vel"]
        #    ),0,ddof=1)*delta_time**2
        self.pos_variance["linear_pos"] += np.array((0.0,0.0,0.0))*delta_time**2
        self.pos_variance["angular_pos"] += np.array((.05,.05,.05))*delta_time**2

        '''np.var((
            self.prev_odo["angular_vel"],
            self.latest_odo["angular_vel"],
            self.latest_twist["angular_vel"]
        ),0,ddof=1)*delta_time**2'''
        
        #self.latest_vision["linear_vel"] = (0.0,0.0,0.0)

    def step(self):
        """Step the integrator

        Pops the integral value for the caller to use,
        resets the integral internally to be ready for the
        next step.

        Returns:
            The stored integrals (position and variance) since last step.

        """
        self.update_integral(time.time())

        self.prev_access_stamp = self.latest_access_stamp
        self.latest_access_stamp = time.time()
        
        _last_access_pos_integral = copy.deepcopy(self.last_access_pos_integral)
#        _last_access_pos_integral["linear_pos"] = self.last_access_pos_integral["linear_pos"]
#        _last_access_pos_integral["angular_pos"] = self.last_access_pos_integral["angular_pos"]
        self.last_access_pos_integral = copy.deepcopy(self.pos_integral)
#        self.last_access_pos_integral["linear_pos"] = self.pos_integral["linear_pos"]
#        self.last_access_pos_integral["angular_pos"] = self.pos_integral["angular_pos"]

        _pos_integral = {
            "linear_pos":self.pos_integral["linear_pos"] - _last_access_pos_integral["linear_pos"],
            "angular_pos":self.pos_integral["angular_pos"] - _last_access_pos_integral["angular_pos"]
        }
        
        _pos_variance = self.pos_variance.copy()
        self.pos_variance = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}

        return _pos_integral, _pos_variance



if __name__ == "__main__":
    options = {
            "resample_interval": 1e6,
            "move_interval": .5,
            "measure_interval": 2,
            "weight_interval": 4,
            "publish_interval": 1e6,
            "resample_noise_count": 0,
            "num_particles": 30
        }
    options = pf.FilterOptions(options)
    robot_filter = RobotLocalizer(options)
    robot_filter.start()