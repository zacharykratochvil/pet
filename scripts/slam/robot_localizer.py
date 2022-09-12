"""ROS node for localization as part of a broader SLAM package

This module integrates sensor data and motor commmands from the robotic pet
into a particle filter for robot localization.

Typical usage example:

Call as part of a launch file that also starts several ROS node dependencies
such as the mapper and sensor nodes.

Copyright Zachary Kratochvil, 2022. All rights reserved.

"""
from particle_filter import ParticleFilter
import rospy
import cv2
import time
import numpy as np

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
        self.drive_sub = rospy.Subscriber("speech_audio", StampedAudio, self.audio_cb, queue_size=2)
        self.witmotion_sub = rospy.Subscriber("imu", sensor_msgs/Imu, self.integrator.on_odo, queue_size=2)
        self.camera_sub = rospy.Subscriber("speech_audio", StampedAudio, self.audio_cb, queue_size=2)

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
        self.latest_twist = {"linear_vel":(0,0,0),"angular_vel":(0,0,0)}
        self.latest_odo = {"linear_acc":(0,0,0),"angular_vel":(0,0,0)}
        self.latest_corr_linear_acc = (0,0,0)
        self.latest_stamp = time.time()

        # stores odometer estimated linear velocities
        self.vel_integral_odo = {"linear_vel":np.array([0,0,0])}

        # stores outputs
        self._clear_integral()

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
        prev_odo = self.latest_odo
        prev_corr_linear_acc = self.latest_corr_linear_acc
        prev_stamp = self.latest_stamp
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
        self.latest_stamp = time.time()

        # calculate correction of linear acceleration for gravity
        g = 9.81 # m/s**2
        g_x = g*np.sin(self.pos_integral["angular_pos"][0])
        g_y = g*np.sin(self.pos_integral["angular_pos"][1])
        g_z = g - g_x - g_y
        g_correction = (g_x, g_y, g_z)

        self.latest_corr_linear_acc = self.latest_odo["linear_acc"] - g_correction

        # update integrals
        delta_time = self.latest_stamp - prev_stamp
        prev_vel_integral_odo = self.vel_integral_odo
        self.vel_integral_odo["linear_vel"] += np.mean((
                prev_corr_linear_acc,
                self.latest_corr_linear_acc
            ))*delta_time

        self.pos_integral["linear_pos"] += np.mean((
                prev_vel_integral_odo["linear_pos"],
                self.vel_integral_odo["linear_pos"]
            ))*delta_time
        self.pos_integral["angular_pos"] += np.mean((
                prev_odo["angular_vel"],
                self.latest_odo["angular_vel"]
            ))*delta_time

        self.pos_variance["linear_pos"] += np.var((
                prev_vel_integral_odo["linear_pos"],
                self.vel_integral_odo["linear_pos"],
                self.latest_twist["linear_pos"]
            ))*delta_time**2
        self.pos_variance["angular_pos"] += np.var((
                prev_odo["angular_vel"],
                self.latest_odo["angular_vel"],
                self.latest_twist["angular_vel"]
            ))*delta_time**2

    def step(self):
        """Step the integrator

        Pops the integral value for the caller to use,
        resets the integral internally to be ready for the
        next step.

        Returns:
            The stored integrals (position and variance).

        """
        _pos_integral = self.pos_integral
        _pos_variance = self.pos_variance
        self._clear_integral()
        return _pos_integral, _pos_variance

    def _clear_integral(self):
        """Sets the stored integral values to zero

        """
        self.pos_integral = {"linear_pos":np.array([0,0,0]),"angular_pos":np.array([0,0,0])}
        self.pos_variance = {"linear_pos":np.array([0,0,0]),"angular_pos":np.array([0,0,0])}


if __name__ == "__main__":
    pass