#!/usr/bin/python
from robot_localizer import RobotLocalizer
import time

from sensor_msgs.msg import Imu


if __name__ == "__main__":

    # testing
    robot_filter = RobotLocalizer()
    
    odo_msg = Imu()
    
    odo_msg.linear_acceleration.x = 0
    odo_msg.linear_acceleration.y = 0
    odo_msg.linear_acceleration.z = 9.85
    odo_msg.angular_velocity.x = 0
    odo_msg.angular_velocity.y = 0
    odo_msg.angular_velocity.z = 0
    robot_filter.integrator.on_odo(odo_msg)

    time.sleep(.01)

    odo_msg.linear_acceleration.x = 0
    odo_msg.linear_acceleration.y = 0
    odo_msg.linear_acceleration.z = 9.79
    odo_msg.angular_velocity.x = 0
    odo_msg.angular_velocity.y = 0
    odo_msg.angular_velocity.z = 0
    robot_filter.integrator.on_odo(odo_msg)

    time.sleep(.01)

    odo_msg.linear_acceleration.x = 0
    odo_msg.linear_acceleration.y = -1
    odo_msg.linear_acceleration.z = 9.82
    odo_msg.angular_velocity.x = 0
    odo_msg.angular_velocity.y = 0
    odo_msg.angular_velocity.z = 0
    robot_filter.integrator.on_odo(odo_msg)

    time.sleep(1)

    odo_msg.linear_acceleration.x = 0
    odo_msg.linear_acceleration.y = 0
    odo_msg.linear_acceleration.z = 9.81
    odo_msg.angular_velocity.x = 0
    odo_msg.angular_velocity.y = 0
    odo_msg.angular_velocity.z = 1
    robot_filter.integrator.on_odo(odo_msg)

    time.sleep(1)

    print(robot_filter.integrator.step())
