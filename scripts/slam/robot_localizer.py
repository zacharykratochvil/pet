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
import numpy as np
import threading

from freenove_ros.msg import TwistDuration, SensorDistance
from sensor_msgs.msg import Imu, PointCloud
from geometry_msgs.msg import Point32

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
        options.options["initial_linear_dist"] = pf.ZeroDistribution2D()
        super().__init__(options)
        
        self.integrator = Integrator()
        self.accumulator = Accumulator()
        self.latest_map = None

        self.move_timer = threading.Timer(self.options["move_interval"], self.move)
        self.weight_timer = threading.Timer(self.options["weight_interval"], self.weight)
        self.publish_timer = threading.Timer(self.options["publish_interval"], self.publish_particles)

        self.drive_sub = rospy.Subscriber("drive_twist", TwistDuration, self.integrator.on_twist, queue_size=1)
        self.witmotion_sub = rospy.Subscriber("imu", Imu, self.integrator.on_odo, queue_size=100)
        self.ultra_sub = rospy.Subscriber("ultrasonic_distance", SensorDistance, self.accumulator.on_ultra, queue_size=1)
        self.map_sub = rospy.Publisher("map", PointCloud, self.map, queue_size=1)

        self.measurement_pub = rospy.Publisher("measured_particles", PointCloud, queue_size=1)
        self.particle_pub = rospy.Publisher("robot_particles", PointCloud, queue_size=1)

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
        self.publish_timer.start()

        rospy.spin()

    def map(self, data):
        self.map = self.decloud(data)
        
    def move(self):
        """Connects to the next available port.

        Args:
            minimum: A port value greater or equal to 1024.

        Returns:
            The new minimum port.

        """
        # restart timer
        self.move_timer.cancel()
        self.move_timer = threading.Timer(self.options["move_interval"], self.move)
        self.move_timer.start()

        # only proceed if can obtain lock
        if self.locked == True:
            return False
        else:
            self.locked = True

        delta_pos, var = self.integrator.step()

        # produce a list of random errors to apply to particles
        _args_list = (delta_pos["linear_pos"][0], delta_pos["linear_pos"][1], var["linear_pos"][0], var["linear_pos"][1])
        lin_error_dist = pf.GaussianDistribution2D(*_args_list)
        lin_errors = lin_error_dist.draw(self.options["num_particles"])
        lin_errors = np.hstack([lin_errors, np.zeros([self.options["num_particles"],1])])

        ang_errors = np.random.normal(delta_pos["angular_pos"][2],var["angular_pos"][2],self.options["num_particles"])

        # apply deltas to particles
        weights = np.zeros([self.options["num_particles"],1])
        particle_deltas = np.hstack([lin_errors[:,0:2], np.reshape(ang_errors,[-1,1]), weights])
        self.particles += particle_deltas
        self.particles[:,self.ANGLE] = ((self.particles[:,self.ANGLE] + 180) % 360) - 180

        self.locked = False
        return True

    def weight(self):
        """Handles measurement inputs.

        Args:
            

        Returns:
            

        """
        # restart timer
        self.weight_timer.cancel()
        self.weight_timer = threading.Timer(self.options["weight_interval"], self.weight)
        self.weight_timer.start()

        # generate list of points, convert cm to m
        distances = np.array(self.accumulator.get_data())/100
        #rospy.loginfo(distances)
        scatter = [np.random.normal(0,np.abs(np.tan(15/180*np.pi)*dist)) for dist in distances]
        xy = np.hstack([np.reshape(distances,[-1,1]), np.reshape(scatter,[-1,1])])
        particles = np.hstack([xy, np.zeros([np.shape(xy)[0], 2])])

        # publish points
        pc = self.make_cloud(particles)
        self.measurement_pub.publish(pc)

        # compare to map, score
        for i in range(self.options["num_particles"]):
            pass


    def publish_particles(self):
        self.publish_timer.cancel()
        self.publish_timer = threading.Timer(self.options["publish_interval"], self.publish_particles)
        self.publish_timer.start()

        pc = self.make_cloud(self.particles)
        self.particle_pub.publish(pc)


class Accumulator:
    def __init__(self):
        self._ultrasonic_data = []

    def on_ultra(self, data):
        if data.cm != -1:
            self._ultrasonic_data.append(data.cm)

    def get_data(self):
        _temp = self._ultrasonic_data
        self._ultrasonic_data = []
        return _temp


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
        self.latest_odo = {"linear_acc":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.latest_corr_linear_acc = (0.0,0.0,0.0)
        self.latest_store_stamp = time.time()
        self.latest_access_stamp = time.time()

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

    def _clear_twist(self):
        self.update_integral(time.time())
        self.latest_twist = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}

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

    def on_odo(self, msg):
        """Processes and integrates Imu messages.

        Args:
            msg: the Imu message received by the subscriber to imu

        """
        # store previous and update latest
        self.prev_odo = self.latest_odo
        self.latest_odo["linear_acc"] = (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            )
        self.latest_odo["angular_vel"] = (
                msg.angular_velocity.x*180/np.pi,
                msg.angular_velocity.y*180/np.pi,
                msg.angular_velocity.z*180/np.pi
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
            x_percent_g = max(-1.0,min(1.0,self.latest_odo["linear_acc"][1]/g))
            y_percent_g = max(-1.0,min(1.0,self.latest_odo["linear_acc"][0]/g))
            x = np.arcsin(x_percent_g)
            y = -np.arcsin(y_percent_g)

            self.pos_integral["angular_pos"] = (x, y, self.pos_integral["angular_pos"][2])
            _pos_integral["angular_pos"] = self.pos_integral["angular_pos"]

        g_x = -g*np.sin(_pos_integral["angular_pos"][1]/180*np.pi)
        g_y = g*np.sin(_pos_integral["angular_pos"][0]/180*np.pi)
        g_z = -np.sqrt(g**2 - g_x**2 - g_y**2)
        g_correction = np.asarray((g_x, g_y, g_z))

        self.prev_corr_linear_acc = self.latest_corr_linear_acc
        if np.sum(np.abs(self.latest_odo["angular_vel"])) == 0:
            self.latest_corr_linear_acc = (0.0, 0.0, 0.0)
            self.vel_integral_odo = {"linear_vel":(0.0, 0.0, 0.0)}
        else:
            self.latest_corr_linear_acc = self.latest_odo["linear_acc"] - g_correction

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
        prev_vel_integral_odo = self.vel_integral_odo
        self.vel_integral_odo["linear_vel"] += np.mean((
                self.prev_corr_linear_acc,
                self.latest_corr_linear_acc
            ),0)*delta_time
        for i in range(3):
            _vel_pointer = self.vel_integral_odo["linear_vel"]
            if np.abs(_vel_pointer[i]) > self.VELOCITY_MAX:
                _vel_pointer[i] = np.sign(_vel_pointer[i])*self.VELOCITY_MAX

        self.pos_integral["linear_pos"] += np.mean((
                prev_vel_integral_odo["linear_vel"],
                self.vel_integral_odo["linear_vel"],
                self.latest_twist["linear_vel"]
            ),0)*delta_time
        self.pos_integral["angular_pos"] += self._calc_angular_vel()*delta_time
        self.pos_integral["angular_pos"] = ((self.pos_integral["angular_pos"] + 540) % 360) - 180

        self.pos_variance["linear_pos"] += np.var((
                prev_vel_integral_odo["linear_vel"],
                self.vel_integral_odo["linear_vel"],
                self.latest_twist["linear_vel"]
            ),0,ddof=1)*delta_time**2
        self.pos_variance["angular_pos"] += np.array((.05,.05,.05))*delta_time**2

        '''np.var((
            self.prev_odo["angular_vel"],
            self.latest_odo["angular_vel"],
            self.latest_twist["angular_vel"]
        ),0,ddof=1)*delta_time**2'''

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
        
        _last_access_pos_integral = self.last_access_pos_integral.copy()
#        _last_access_pos_integral["linear_pos"] = self.last_access_pos_integral["linear_pos"]
#        _last_access_pos_integral["angular_pos"] = self.last_access_pos_integral["angular_pos"]
        self.last_access_pos_integral = self.pos_integral.copy()
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
            "resample_interval": .5,
            "move_interval": .1,
            "weight_interval": .5,
            "publish_interval": 1,
            "resample_noise_count": 0
        }
    options = pf.FilterOptions(options)
    robot_filter = RobotLocalizer(options)
    robot_filter.start()