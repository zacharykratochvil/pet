"""Particle Filter Core Tools

This module defines the basic structure of the particle filter.

Typical usage example:

Not yet defined.

Copyright Zachary Kratochvil, 2022. All rights reserved.

"""
import numpy as np
import threading
import time

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
            "null_dist": GaussianDistribution2D(xmean=0,ymean=0,xstd=1,ystd=1),
            "num_particles": 1000,
            "resample_interval": .05,
            "resample_noise_count": 5,
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

        # initialize particle list
        self.particles = self.init_particles(self.options["num_particles"])

        # initiate resampling process
        self.resampling = False
        self.resample_timer = None
        self.reset_resample_timer()

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
            particles_xy = self.options["null_dist"].draw(n)
            particles_angle = np.random.uniform(-180, 180, n)
            particles_weight = np.ones((n, self.options["initial_weight"]))
            return np.hstack((particles_xy, particles_angle[:,np.newaxis], particles_weight))

    def reset_resample_timer(self):
        """Resets the resample timer

        Returns:
            True if the previous timer had completed successfully
            False if the previous timer had to be interrupted

        """

        # determines whether previous timer is running
        interrupting = False
        if self.resample_timer is not None and self.resample_timer.is_alive():
            self.resample_timer.cancel()
            interrupting = True

        # starts new timer
        self.resample_timer = threading.Timer(interval=self.options["resample_interval"], function=self.resample)
        self.resample_timer.start()

        return not interrupting

    def resample(self):
        """Resamples particles by weight

        Returns:
            True if resampling completed successfully
            False if canceled because previous resampling was still running

        """
        
        # fail if not done processing previous request yet
        if self.resampling == True:
            self.reset_resample_timer()
            return False
        
        # otherwise, resample the particles by weight
        self.resampling = True
        self.reset_resample_timer()

        current_num_particles = np.shape(self.particles)[0]

        cumulative_importance = 0
        importance_cdf = np.zeros(current_num_particles, dtype=float)
        for i in range(current_num_particles):
            cumulative_importance += self.particles[i,3]
            importance_cdf[i] = cumulative_importance

        # draw requested number of particles from this filter's learned distribution
        num_from_filter_dist = self.options["num_particles"] - self.options["resample_noise_count"]
        new_particle_inds = np.zeros(num_from_filter_dist, dtype=int)
        for i in range(num_from_filter_dist):
            selection = np.random.uniform(0,importance_cdf[-1])
            new_particle_inds[i] = np.searchsorted(importance_cdf, selection)
        self.particles = self.particles[new_particle_inds,:]
        self.particles[:,3] = self.options["initial_weight"]

        # draw remaining particles from this filter's null distribution
        if self.options["resample_noise_count"] > 0:
            null_particles = self.init_particles(self.options["resample_noise_count"])
            self.particles = np.vstack((self.particles, null_particles))

        self.resampling = False
        return True
    
    def get_occupancy_grid(self):
        raise Exception("get_occupancy_grid function is not yet defined")

    def close(self):
        self.resample_timer.cancel()
        time.sleep(.001)
        self.resample_timer.cancel()


if __name__ == "__main__":
    print("Starting unit tests for slam.py")
    test_filter_1 = ParticleFilter()
    print("first 10 particles: " + str(test_filter_1.particles[1:10,:]))
    time.sleep(5)
    print("first 10 particles: " + str(test_filter_1.particles[1:10,:]))
    time.sleep(5)
    print("first 10 particles: " + str(test_filter_1.particles[1:10,:]))
    test_filter_1.close()