#!/usr/bin/python
"""ROS node for localization as part of a broader SLAM package

This module integrates sensor data and motor commmands from the robotic pet
into a particle filter for robot localization.

Typical usage example:

Call as part of a launch file that also starts several ROS node dependencies
such as the mapper and sensor nodes.

Copyright Zachary Kratochvil, 2022. All rights reserved.

"""
from common.particle_filter import ParticleFilter
import rospy
import cv2
import time
import numpy as np

from freenove_ros.msg import TwistDuration
from sensor_msgs.msg import Imu

class RobotLocalizer(ParticleFilter):
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
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
        self.integrator = Integrator()
        #self.drive_sub = rospy.Subscriber("speech_audio", StampedAudio, self.audio_cb, queue_size=2)
        self.witmotion_sub = rospy.Subscriber("imu", Imu, self.integrator.on_odo, queue_size=2)
        #self.camera_sub = rospy.Subscriber("speech_audio", StampedAudio, self.audio_cb, queue_size=2)

    def start(self):
        rospy.init_node("robot_localizer", anonymous=False)
        #rospy.spin()
        while True:
            time.sleep(5)
            rospy.loginfo(self.integrator.step())

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
        # stores last measurements
        self.latest_twist = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.latest_odo = {"linear_acc":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.latest_corr_linear_acc = (0.0,0.0,0.0)
        self.latest_store_stamp = time.time()
        self.latest_access_stamp = time.time()

        self.prev_twist = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
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

    def on_twist(self, msg):
        """Connects to the next available port.

        Args:
            minimum: the TwistDuration message

        """
        prev_twist = self.latest_twist

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
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            )

        # calculate correction of linear acceleration for gravity
        delta_time = time.time() - self.latest_store_stamp
        _pos_integral = {
                "angular_pos": self.pos_integral["angular_pos"] +
                                self._calc_angular_vel()*delta_time
            }

        g = -9.81 # m/s**2
        if np.sum(np.abs(self.latest_odo["angular_vel"])) == 0:
            x_percent_g = max(-1.0,min(1.0,self.latest_odo["linear_acc"][1]/g))
            y_percent_g = max(-1.0,min(1.0,self.latest_odo["linear_acc"][0]/g))
            x = np.arcsin(x_percent_g)
            y = -np.arcsin(y_percent_g)

            self.pos_integral["angular_pos"] = (x, y, self.pos_integral["angular_pos"][2])
            _pos_integral["angular_pos"] = self.pos_integral["angular_pos"]

        g_x = -g*np.sin(_pos_integral["angular_pos"][1])
        g_y = g*np.sin(_pos_integral["angular_pos"][0])
        g_z = -np.sqrt(g**2 - g_x**2 - g_y**2)
        g_correction = np.asarray((g_x, g_y, g_z))

        self.prev_corr_linear_acc = self.latest_corr_linear_acc
        if np.sum(np.abs(self.latest_odo["angular_vel"])) == 0:
            self.latest_corr_linear_acc = (0.0, 0.0, 0.0)
            self.vel_integral_odo = {"linear_vel":(0.0, 0.0, 0.0)}
        else:
            self.latest_corr_linear_acc = self.latest_odo["linear_acc"] - g_correction

        # calculate position integrals
        self.update_integral()

    def _calc_angular_vel(self):
        return np.mean((
                    self.prev_odo["angular_vel"],
                    self.latest_odo["angular_vel"],
                    #self.prev_twist["angular_vel"]
                ),0)

    def update_integral(self):
        # update time
        self.prev_store_stamp = self.latest_store_stamp
        self.latest_store_stamp = time.time()
        delta_time = self.latest_store_stamp - self.prev_store_stamp

        # update integrals
        prev_vel_integral_odo = self.vel_integral_odo
        self.vel_integral_odo["linear_vel"] += np.mean((
                self.prev_corr_linear_acc,
                self.latest_corr_linear_acc
            ),0)*delta_time

        self.pos_integral["linear_pos"] += np.mean((
                prev_vel_integral_odo["linear_vel"],
                self.vel_integral_odo["linear_vel"],
                self.prev_twist["linear_vel"]
            ),0)*delta_time
        self.pos_integral["angular_pos"] += self._calc_angular_vel()*delta_time

        self.pos_variance["linear_pos"] += np.var((
                prev_vel_integral_odo["linear_vel"],
                self.vel_integral_odo["linear_vel"],
                self.prev_twist["linear_vel"]
            ),0,ddof=1)*delta_time**2
        self.pos_variance["angular_pos"] += np.var((
                self.prev_odo["angular_vel"],
                self.latest_odo["angular_vel"],
                self.prev_twist["angular_vel"]
            ),0,ddof=1)*delta_time**2

    def step(self):
        """Step the integrator

        Pops the integral value for the caller to use,
        resets the integral internally to be ready for the
        next step.

        Returns:
            The stored integrals (position and variance) since last step.

        """
        self.update_integral()

        self.prev_access_stamp = self.latest_access_stamp
        self.latest_access_stamp = time.time()
        
        _last_access_pos_integral = self.last_access_pos_integral.copy()
        _last_access_pos_integral["linear_pos"] = self.last_access_pos_integral["linear_pos"].copy()
        _last_access_pos_integral["angular_pos"] = self.last_access_pos_integral["angular_pos"].copy()
        self.last_access_pos_integral = self.pos_integral.copy()
        self.last_access_pos_integral["linear_pos"] = self.pos_integral["linear_pos"].copy()
        self.last_access_pos_integral["angular_pos"] = self.pos_integral["angular_pos"].copy()

        _pos_integral = {
            "linear_pos":self.pos_integral["linear_pos"] - _last_access_pos_integral["linear_pos"],
            "angular_pos":self.pos_integral["angular_pos"] - _last_access_pos_integral["angular_pos"]
        }
        
        _pos_variance = self.pos_variance.copy()
        self.pos_variance = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}

        return _pos_integral, _pos_variance

if __name__ == "__main__":
    robot_filter = RobotLocalizer()
    robot_filter.start()