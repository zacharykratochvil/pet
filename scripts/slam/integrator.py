import common.particle_filter as pf
import rospy
import cv2
import time
import copy
import numpy as np
import threading
from scipy.spatial import transform

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
        self.latest_odo = {"linear_acc":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0),"orientation":(0.0,0.0,0.0)}
        self.latest_corr_linear_acc = (0.0,0.0,0.0)
        self.latest_store_stamp = time.time()
        self.latest_access_stamp = time.time()
        self.latest_distances = None

        self.prev_vision = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.prev_odo = {"linear_acc":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0),"orientation":(0.0,0.0,0.0)}
        self.prev_corr_linear_acc = (0.0,0.0,0.0)
        self.prev_store_stamp = time.time()
        self.prev_access_stamp = time.time()
        self.prev_store_stamp_odo = self.latest_store_stamp
        self.latest_store_stamp_odo = time.time()

        self.initial_orientation = {"orientation":(0.0,0.0,0.0)}

        # stores odometer estimated accelerations and velocities
        self.vel_integral_odo = {"linear_vel":np.array([0.0,0.0,0.0])}
        
        self.odo_counter = 0
        self.odo_downsample_ratio = 1
        self.pre_filtered_odo = {"linear_acc":np.zeros([self.odo_downsample_ratio,3]),"angular_vel":np.zeros([self.odo_downsample_ratio,3])}
        self.pre_filtered_timestamps = np.empty([self.odo_downsample_ratio,3])
        self.pre_filtered_timestamps[:] = np.nan
        self.filtered_odo = {"linear_acc":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}

        # stores filtered velocity
        self.velocity_filter_size = 20
        self.velocity_filter = np.zeros([self.velocity_filter_size,3+3+1+3])#slots,xyz,xyzweight,timestamp,variance
        #self.velocity_weight = np.ones([self.velocity_filter_size,3])

        # zero integrals
        self.pos_integral = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}
        self.last_access_pos_integral = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}
        self.pos_variance = {"linear_pos":np.array([0.0,0.0,0.0]),"angular_pos":np.array([0.0,0.0,0.0])}

        # twist callback        
        self.twist_timer = threading.Timer(0, self._clear_twist)

        # integral update timer
        self.update_timer = threading.Timer(.01, self.update_pos_integral)
        self.update_timer.start()


    def update_distances(self, distances):
        self.latest_distances = distances

    def _clear_twist(self):
        self.latest_twist = {"linear_vel":(0.0,0.0,0.0),"angular_vel":(0.0,0.0,0.0)}
        self.update_velocity(self.latest_twist, weight=(5,5,5), stamp=time.time())

    #average with heavier weight to lower value
    def _smooth(self, array1, array2, weight_ratio=5):
        min_index = np.argmin([array1, array2], axis = 0)

        weights = np.ones([2,3])
        for i in range(len(min_index)):
            weights[min_index[i],i] = weight_ratio

        return np.average([array1, array2], weights=weights, axis = 0)

    def on_twist(self, msg):
        """Processes and integrates Twist drive request messages.

        Args:
            msg: the TwistDuration message received by the subscriber
            
        """
        # update timers
        self.twist_timer.cancel()
        #duration = msg.duration.secs + msg.duration.nsecs*1e-9
        self.twist_timer = threading.Timer(msg.duration, self._clear_twist)
        self.twist_timer.start()

        # update twists
        self.latest_twist["linear_vel"] = (
                msg.velocity.linear.x,
                msg.velocity.linear.y/2,
                msg.velocity.linear.z
            )
        self.latest_twist["angular_vel"] = (
                msg.velocity.angular.x,
                msg.velocity.angular.y,
                msg.velocity.angular.z
            )

        # update integrals
        #self.update_velocity(self.latest_twist, weight=(10,10,10), stamp=time.time())

    def on_odo(self, msg):
        """Processes and integrates Imu messages.

        Args:
            msg: the Imu message received by the subscriber to imu

        """

        # store previous and update latest
        self.prev_odo = copy.deepcopy(self.latest_odo)
        self.latest_odo["angular_vel"] = (
                -msg.angular_velocity.x*180/np.pi,
                msg.angular_velocity.y*180/np.pi,
                -msg.angular_velocity.z*180/np.pi
            )
        rotation_obj = transform.Rotation.from_quat((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))
        if np.sum(self.initial_orientation["orientation"]) == 0 and np.sum(self.prev_odo["orientation"]) == 0:
            self.initial_orientation["orientation"] = -rotation_obj.as_euler('XYZ',degrees=True)
        self.latest_odo["orientation"] = -rotation_obj.as_euler('XYZ',degrees=True) - self.initial_orientation["orientation"]

        rospy.loginfo(self.latest_odo["angular_vel"][2])

        stamp = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9

        self.pre_filtered_timestamps[1:,:] = self.pre_filtered_timestamps[:-1,:]
        self.pre_filtered_timestamps[0,:] = stamp
        self.pre_filtered_odo["linear_acc"][1:,:] = self.pre_filtered_odo["linear_acc"][:-1,:]
        self.pre_filtered_odo["linear_acc"][0,:] = self.latest_odo["linear_acc"]
        self.pre_filtered_odo["angular_vel"][1:,:] = self.pre_filtered_odo["angular_vel"][:-1,:]
        self.pre_filtered_odo["angular_vel"][0,:] = self.latest_odo["angular_vel"]
        
        self.odo_counter += 1
        if self.odo_counter >= self.odo_downsample_ratio:
            self.filter_odo()

    def update_pos_integral(self):
        """Performs most of the integration logic.

        Should be called anytime any of the integrator's inputs are updated.

        """

        # update timer
        self.update_timer.cancel()
        self.update_timer = threading.Timer(.01, self.update_pos_integral)
        self.update_timer.start()

        # update time
        self.prev_store_stamp = self.latest_store_stamp
        self.latest_store_stamp = time.time()
        delta_time = self.latest_store_stamp - self.prev_store_stamp

        
        #rospy.loginfo("prev_integral: " + str(prev_vel_integral_odo["linear_vel"]))
        #rospy.loginfo(self.vel_integral_odo["linear_vel"])
        #rospy.loginfo(self.latest_twist["linear_vel"])

        linear_velocity = self.transform_one(np.asarray(self.latest_twist["linear_vel"]), np.asarray([0,0,self.pos_integral["angular_pos"][2]]))[0:3]
        linear_variance = np.asarray([.1,.1,.1])#self.get_velocity()
        self.pos_integral["linear_pos"] += linear_velocity*delta_time
            #np.nanmean((
                #prev_vel_integral_odo["linear_vel"],
                #self.vel_integral_odo["linear_vel"],
                #self.latest_vision["linear_vel"],
                #self.latest_vision["linear_vel"],
                #self.prev_vision["linear_vel"],
                #self.latest_vision["linear_vel"]
                #self.latest_twist["linear_vel"]
                #self.latest_twist["linear_vel"]
            #), axis = 0)*delta_time

        #rospy.loginfo("integral: " + str(self.pos_integral["linear_pos"]))
        self.pos_integral["angular_pos"] += self._calc_angular_vel()*delta_time
        self.pos_integral["angular_pos"] = ((self.pos_integral["angular_pos"] + 540) % 360) - 180

        #rospy.loginfo(self.pos_integral["angular_pos"])

        #self.pos_variance["linear_pos"] += np.nanvar((
        #        prev_vel_integral_odo["linear_vel"],
        #        self.vel_integral_odo["linear_vel"],
        #        self.latest_vision["linear_vel"]
                #self.latest_twist["linear_vel"]
                #self.latest_twist["linear_vel"]
        #    ),0,ddof=1)*delta_time**2
        self.pos_variance["linear_pos"] += linear_variance*delta_time**2
        self.pos_variance["angular_pos"] += np.array((1,1,1))*delta_time**2 #np.array((.05,.05,.05))*delta_time**2

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
        self.update_pos_integral()

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

        #rospy.loginfo(_pos_integral)

        return _pos_integral, _pos_variance

    def get_orientation(self):
        return ((self.latest_odo["orientation"] + 540) % 360) - 180

    def transform_one(self, point, reference):
        '''
        Transforms point (x,y,angle) into the reference frame of one particle.
        '''
        X = 0
        Y = 1
        ANGLE = 2

        #rotate linear coords
        angle_1 = np.pi/180*reference[ANGLE]
        length = np.sqrt(point[X]**2 + point[Y]**2)
        angle_2 = np.arctan2(point[Y],point[X])

        x = np.cos(angle_1 + angle_2)*length + reference[X]
        y = np.sin(angle_1 + angle_2)*length + reference[Y]
        angle = 180/np.pi*(angle_1 + angle_2) + point[ANGLE]
        angle = ((angle + 540) % 360) - 180

        return np.array((x,y,angle,1))

