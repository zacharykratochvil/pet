"""Particle Filter Core Tools

This module defines the basic structure of the particle filter.

Typical usage example:

Not yet defined.

Copyright Zachary Kratochvil, 2022. All rights reserved.

"""
import numpy as np
import threading
import traceback
import inspect
import time
import rospy
import gc
import copy
from functools import wraps

from sensor_msgs.msg import PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

class Distribution2D:
    def zip_coords(self, xs, ys):
        """zip_coords merges two lists into a list of tuples
        
        Args:
            xs: list of x coords
            ys: list of y coords

        Returns:
            The zipped coordinate list.

        """
        return np.hstack((xs[:,np.newaxis], ys[:,np.newaxis]))

class UniformDistribution2D(Distribution2D):
    """Defines a uniform distribution in 2D

    Attributes:
        xlim: min and max of distribution in x dimension
        ylim: min and max of distribution in y dimension

    """
    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim

    def draw(self, n):
        """Draws n numbers from the distribution
        """
        xs = np.random.uniform(*self.xlim, n)
        ys = np.random.uniform(*self.ylim, n)
        return super().zip_coords(xs, ys)

class GaussianDistribution2D(Distribution2D):
    """Defines a gaussian distribution in 2D

    Attributes:
        xmean: mean in the x dimension
        ymean: mean in the y dimension
        xstd: standard deviation in the x dimension
        ystd: standard deviation in the y dimension
    
    """
    def __init__(self, xmean, ymean, xstd, ystd):
        self.xmean = xmean
        self.xstd = xstd
        self.ymean = ymean
        self.ystd = ystd

    def draw(self, n):
        """Draws n numbers from the distribution
        """
        
        xs = np.random.normal(self.xmean, self.xstd, n)
        ys = np.random.normal(self.ymean, self.ystd, n)
        return super().zip_coords(xs, ys)

class ZeroDistribution2D():
    """Defines a trivial empty distribution

    This distribution will always return zero.

    """
    def __init__(self):
        pass

    def draw(self, n):
        return np.zeros([n,2])

class EmptyDistribution():
    """Defines a trivial empty distribution

    This distribution will always return the empty set.

    """
    def __init__(self):
        pass

    def draw(self, n):
        return []

class FilterOptions:
    """Options object stores options for creating a particle filter

    Attributes:

        options: a dictionary of options

                null_dist: The distribution object from which to draw
                    new (including initial) random particle positions.
                initial_linear_dist: Set the initial distribution independent
                    of the null_distribution.
                initial_angular_dist: Set the initial distribution independent
                    of the null_distribution.
                num_particles: The number of particles to initialize
                    the filter with and resample to.
                resample_interval: The interval between calls to the
                    resample method (in seconds).
                resample_noise_count: Number of added particles that
                    should be noise
                initial_weight:
    
    """

    def __init__(self, options={}):
        # set default options
        self.options = {
            "null_linear_dist": UniformDistribution2D((-10,10),(-10,10)),
            "null_angular_dist": lambda x: np.random.uniform(-180,180,x),
            "initial_linear_dist": ZeroDistribution2D(),
            "initial_angular_dist": lambda x: np.zeros(x),
            "num_particles": 2000,
            "resample_interval": 1e6,
            "use_timers": True,
            "resample_noise_count": 3,
            "reset_weight_on_resample": True,
            "initial_weight": 1
        }

        # override with user preferences
        self.options.update(options)


class ParticleFilter:
    """Parent class of any 2D particle filter like object

    This class is abstract and cannot be instantiated on its own,
    but must be inherited by another class.

    Attributes:
        options: options object with list of options, see options class
        particles: list of (x,y,angle,weight) tuples
        resample_timer: timer to initiate resamplings

    """
    X = 0
    Y = 1
    ANGLE = 2
    WEIGHT = 3

    def __init__(self, options=FilterOptions()):
        """Constructor for ParticleFilter objects
        
        Args:
            options: options object with list of options, see options class

        Returns:
            The new ParticleFilter object.

        """
        # store arguments
        self.options = options.options

        # validate arguments
        if self.options["num_particles"] <= 0 or self.options["num_particles"] != int(self.options["num_particles"]):
            raise Exception("Particle count must be a positive integer, not " + str(self.options["num_particles"]))
        elif self.options["resample_noise_count"] < 0:
            raise Exception("You cannot add negative particles.")
        elif self.options["resample_noise_count"]/self.options["num_particles"] > 1:
            raise Exception("You cannot add more noise particles than particles total.")
        elif self.options["resample_noise_count"]/self.options["num_particles"] > .8:
            raise Warning("You are adding more than 80% noise at each resampling, this is not recommended.")

        # establish aggressive garbage collection
        #gc.set_threshold(1000,5)

        # initialize particle list
        self.particles = self.init_particles(self.options["num_particles"])
        self.particle_data = np.array([{} for i in range(self.options["num_particles"])])
        self.locked = False

        # initiate resampling process
        self.resampling = False
        if self.options["use_timers"] == True:
            self.resample_timer = threading.Timer(self.options["resample_interval"], self.resample)
            self.resample_timer.start()

        #self.resize_timer = threading.Timer(0, self.resize)

    def init_particles(self, n):
        """Returns particles in null distribution

        Args:
            n: number of particles to generate

        Returns:
            List of n particles

        """
        if n < 1:
            return []
        else:
            if self.options["initial_linear_dist"] is not None:
                particles_xy = self.options["initial_linear_dist"].draw(n)
            else:
                particles_xy = self.options["null_linear_dist"].draw(n)

            if self.options["initial_angular_dist"] is not None:
                particles_angle = self.options["initial_angular_dist"](n)
            else:
                particles_angle = self.options["null_angular_dist"](n)

            particles_weight = np.ones(n)*self.options["initial_weight"]
            return np.hstack((particles_xy, particles_angle[:,np.newaxis], particles_weight[:,np.newaxis]))

    '''
    def reset_resample_timer(self, interval=None):
        """Resets the resample timer

        Returns:
            True if the previous timer had completed successfully
            False if the previous timer had to be interrupted

        """

        if type(interval) == type(None):
            interval = self.options["resample_interval"]

        # determines whether previous timer is running
        interrupting = False
        if self.resample_timer is not None and self.resample_timer.is_alive():
            self.resample_timer.cancel()
            interrupting = True

        # starts new timer
        if self.options["use_timers"] == True:
            self.resample_timer = threading.Timer(interval=interval, function=self.resample)
            self.resample_timer.start()

        return not interrupting


    def decorate(inner_fn, *args, **kwargs):
        
        calling_fn = inspect.stack()[1][3]
        
        def decorator(fn):
            def wrapper(*wrapper_args, **wrapper_kwargs):
                return inner_fn(calling_fn=calling_fn, *args, **kwargs)(fn(*wrapper_args, **wrapper_kwargs))
            
            return wrapper
        return decorator
    '''

    def locking(calling_fn=None, timer_name=None, long_timeout=None, short_timeout=.1, timer_data=None, ignore_lock=False):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                self = args[0]

                if timer_name is not None:
                    exec(f"self.{timer_name}.cancel()")
                    #rospy.logerr(f"got here from {calling_fn}")

                # only proceed if can obtain lock
                if self.locked and not ignore_lock:
                    if timer_name is not None and short_timeout is not None:
                        if type(timer_data) == type(None):
                            exec(f"self.{timer_name} = threading.Timer({short_timeout}, self.{calling_fn})")
                        else:    
                            exec(f"self.{timer_name} = threading.Timer({short_timeout}, self.{calling_fn}, timer_data)")
                        exec(f"self.{timer_name}.start()")
                    return False
                else:
                    if ignore_lock == False:
                        self.locked = True
                    output = None
                    
                    try:

                        if timer_name is not None and long_timeout is not None:
                            exec(f"self.{timer_name} = threading.Timer({long_timeout}, self.{calling_fn})")
                            exec(f"self.{timer_name}.start()")

                        output = fn(self, *args, **kwargs)

                    except:
                        traceback.print_exc()
                    finally:
                        if ignore_lock == False:
                            self.locked = False

                    return output

            return wrapper
        return decorator

    def resample(self, ignore_lock=False):
        """Resamples particles by weight

        Returns:
            True if resampling completed successfully
            False if canceled because previous resampling was still running

        """

        timer_name = "resample_timer" if self.options["use_timers"] == True else None
        @ParticleFilter.locking("resample", timer_name=timer_name, long_timeout=self.options["resample_interval"], ignore_lock=ignore_lock)
        def inner_resample(self, *args, **kwargs):

            weights = self.reweight()
            assert np.sum(weights) > 0, "Weights must be positive."

            current_num_particles = np.shape(self.particles)[0]

            cumulative_importance = 0
            importance_cdf = np.zeros(current_num_particles, dtype=float)
            for i in range(current_num_particles):
                cumulative_importance += weights[i]
                importance_cdf[i] = cumulative_importance

            # draw requested number of particles from this filter's learned distribution
            num_from_filter_dist = self.options["num_particles"] - self.options["resample_noise_count"]
            #rospy.loginfo(num_from_filter_dist)
            new_particle_inds = np.zeros(num_from_filter_dist, dtype=int)
            for i in range(num_from_filter_dist):
                selection = np.random.uniform(0,importance_cdf[-1])
                new_particle_inds[i] = np.searchsorted(importance_cdf, selection)
                
            particles_slice = self.particles[new_particle_inds,:]
            self.particles = copy.deepcopy(particles_slice)
            
            particle_data_slice = self.particle_data[new_particle_inds]
            self.particle_data = np.array([{} for i in range(len(particle_data_slice))])
            for i in range(len(particle_data_slice)):
                self.particle_data[i] = copy.deepcopy(particle_data_slice[i])
            
            if self.options["reset_weight_on_resample"] == True:
                self.particles[:,self.WEIGHT] = self.options["initial_weight"]

            #rospy.loginfo(len(self.particles))

            # draw remaining particles from this filter's null distribution
            if self.options["resample_noise_count"] > 0:
                null_xy = self.options["null_linear_dist"].draw(self.options["resample_noise_count"])
                null_angle = self.options["null_angular_dist"](self.options["resample_noise_count"])
                null_weight = np.ones([self.options["resample_noise_count"], 1])*self.options["initial_weight"]
                null_data = np.array([{} for i in range(self.options["resample_noise_count"])])

                null_particles = np.hstack((null_xy, np.reshape(null_angle, [-1,1]), null_weight))
                self.particles = np.vstack((copy.deepcopy(self.particles), null_particles))
                self.particle_data = np.hstack((copy.deepcopy(self.particle_data), null_data))

            #rospy.loginfo(len(self.particles))

            return True

        inner_resample(self)

    def reweight(self):
        # override to adjust weights before resampling
        return self.particles[:,self.WEIGHT]

    #not supported because it would require passing locks which is not currently supported
    #@ParticleFilter.locking("resize", timer_name="resize_timer")
    #def resize(self, new_size, *args, **kwargs):
    #    self.particles

    def make_cloud(self, points):
        pc = PointCloud()
        pc.header.stamp = rospy.get_rostime()
        pc.header.frame_id = "base_link"
        pc.points = [Point32() for i in range(len(points))]
        pc.channels = [ChannelFloat32(), ChannelFloat32()]
        pc.channels[0].values = np.empty(len(points))
        pc.channels[0].name = 'orientation'
        pc.channels[1].values = np.empty(len(points))
        pc.channels[1].name = 'weight'

        for i, particle in enumerate(points):
            pc.points[i].x = particle[self.X]
            pc.points[i].y = particle[self.Y]
            pc.points[i].z = 0
            pc.channels[0].values[i] = particle[self.ANGLE]
            pc.channels[1].values[i] = particle[self.WEIGHT]
        
        return pc

    def transform_all(self, point):
        '''
        Transforms point (x,y,angle) into the reference frame of each of the particles.
        '''

        tf_particles = np.empty([len(self.particles), 4])
        for i in len(self.particles):
            tf_particles[i] = transform_one(point, self.particles[i])

        return tf_particles

    def transform_one(self, point, reference):
        '''
        Transforms point (x,y,angle) into the reference frame of one particle.
        '''

        #rotate linear coords
        angle_1 = np.pi/180*reference[self.ANGLE]
        length = np.sqrt(point[self.X]**2 + point[self.Y]**2)
        angle_2 = np.arctan2(point[self.Y],point[self.X])

        x = np.cos(angle_1 + angle_2)*length + reference[self.X]
        y = np.sin(angle_1 + angle_2)*length + reference[self.Y]
        angle = point[self.ANGLE]#180/np.pi*(angle_1 + angle_2) + point[self.ANGLE]
        angle = ((angle + 540) % 360) - 180


        return np.array((x,y,angle,1))
    
    def get_occupancy_grid(self):
        raise Exception("get_occupancy_grid function is not yet defined")

    def close(self):
        try:
            self.resample_timer.cancel()
        except:
            pass

    def __deepcopy__(self, memo):
        
        pf = ParticleFilter(FilterOptions(self.options))
        memo[id(self)] = pf
        
        pf.particles = copy.deepcopy(self.particles, memo)

        pf.particle_data = np.array([{} for i in range(len(self.particle_data))])
        memo[id(self.particle_data)] = pf.particle_data
        for i in range(len(pf.particle_data)):
            pf.particle_data[i] = copy.deepcopy(self.particle_data[i], memo)
        
        return pf

    def decloud(cloud):
        points = [(pt.x, pt.y, ang, w) for pt, ang, w in zip(cloud.points, cloud.channels[0].values, cloud.channels[1].values)]
        if len(points) > 0:
            return np.vstack(points)
        else:
            return np.array(())


if __name__ == "__main__":
    print("Starting unit tests for slam.py")
    test_filter_1 = ParticleFilter()
    print("first 10 particles: " + str(test_filter_1.particles[1:10,:]))
    time.sleep(5)
    print("first 10 particles: " + str(test_filter_1.particles[1:10,:]))
    time.sleep(5)
    print("first 10 particles: " + str(test_filter_1.particles[1:10,:]))
    test_filter_1.close()
