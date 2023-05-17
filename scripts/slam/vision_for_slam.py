#!/usr/bin/python
import cv2
#from PIL import Image as PIL_Image
#from io import BytesIO
import numpy as np
import rospy
from common.accumulator import UltraSonicAccumulator
import common.vision_features as vf

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from freenove_ros.msg import SensorDistance

class OpticalFlow:

    def __init__(self):


        self.min_points = 20 #32
        self.min_descriptors = 400 #1000
        self.min_variance = 7.5 #15

        self.accumulator = UltraSonicAccumulator()

        self.camera_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.on_image, queue_size=1)
        self.ultra_sub = rospy.Subscriber("ultrasonic_distance", SensorDistance, self.accumulator.on_ultra, queue_size=1)

        self.prev_points = None
        self.prev_descriptors = None
        self.prev_dots = []

        self.current_distance = 0

        self.vel_pub = rospy.Publisher("optical_velocity", Twist, queue_size = 2)
        

    def start(self):
        rospy.init_node("optical_flow",anonymous=False)
        
        rospy.spin()

    def on_image(self, data):        

        #filter data
        if (rospy.get_rostime() - data.header.stamp).to_sec() > .1:
            return
        
        int_data = np.frombuffer(data.data, np.uint8)
        image = cv2.imdecode(int_data, cv2.IMREAD_COLOR)

        if np.var(image) == 0:
            return

        if data.header.seq % 10 != 0:
            return

        # enhance image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv_image[:,:,2] = clahe.apply(hsv_image[:,:,2])
        self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        #cv2.imshow("image", self.image)
        #cv2.waitKey(1)

        # detect features in data
        cur_points, cur_descriptors = vf.detect(self.image, num_points=128, descriptors_per_point=128)
        if type(self.prev_points) == type(None) or np.shape(self.prev_points)[0] < self.min_points or np.sum(self.prev_descriptors != 0) < self.min_descriptors:
            self.prev_points, self.prev_descriptors = cur_points, cur_descriptors

        #rospy.loginfo(f"curpts: {np.shape(cur_points)}")
        #rospy.loginfo(f"curdsc: {np.shape(cur_descriptors)}")
        #rospy.loginfo(f"prpts: {np.shape(prev_points)}")
        #rospy.loginfo(f"prdxc: {np.shape(prev_descriptors)}")

        # calculate transformation and return no motion if results b
        rospy.loginfo(f"points and descriptors: {np.shape(cur_points)[0]}, {np.sum(cur_descriptors != 0)}, {np.shape(self.prev_points)[0]}, {np.sum(self.prev_descriptors != 0)}")
        if len(cur_points > 2):
            count_rule = np.shape(cur_points)[0] > self.min_points and np.sum(cur_descriptors != 0) > self.min_descriptors and np.shape(self.prev_points)[0] > self.min_points and np.sum(self.prev_descriptors != 0) > self.min_descriptors
            variance_rule = np.all(np.asarray([np.var(cur_points[:,0]), np.var(cur_points[:,1]), np.var(self.prev_points[:,0]), np.var(self.prev_points[:,1])]) > self.min_variance)
        else:
            count_rule = False
            variance_rule = False
        if count_rule and variance_rule:
            translation, zoom, score = vf.estimate_translation_and_zoom(self.prev_points, cur_points, self.prev_descriptors, cur_descriptors, self.image.shape)
        else:
            message = Twist()
            message.linear.x = 0
            message.linear.y = 0
            message.linear.z = 0
            self.vel_pub.publish(message)
            if not count_rule:
                rospy.loginfo("too few observations, discarding")
            elif not variance_rule:
                rospy.loginfo("points are clustered, discarding")

            #self.prev_points = np.copy(cur_points)
            #self.prev_descriptors = np.copy(cur_descriptors)
            return

        # get accurate score
        '''
        width_zoom_correction = 0
        height_zoom_correction = 0
        M = np.asarray([[zoom, 0, (-translation[0]-width_zoom_correction)], [0, zoom, (-translation[1]-height_zoom_correction)]])
        warped = cv2.warpAffine(self.image, M, (np.shape(self.image)[1], np.shape(self.image)[0]))
        warped_points, warped_descriptors = vf.detect(warped, num_points=128, descriptors_per_point=128)
        _, _, score = vf.estimate_translation_and_zoom(self.prev_points, warped_points, self.prev_descriptors, warped_descriptors, self.image.shape)
        '''

        if score < .4 or np.isinf(score) or np.isnan(score) or zoom < .5 or zoom > 2 or np.any(np.abs(translation) > 500):
            message = Twist()
            message.linear.x = 0
            message.linear.y = 0
            message.linear.z = 0
            self.vel_pub.publish(message)
            rospy.loginfo("poor alignment, discarding")

            self.prev_points = np.copy(cur_points)
            self.prev_descriptors = np.copy(cur_descriptors)
            return

        rospy.loginfo(f"zoom: {zoom} translation: {translation} score: {score}")

        # actually calculate flow from transformation and return
        latest_distance = np.array(self.accumulator.get_data())/100
        if len(latest_distance) != 0:
            self.current_distance = np.mean(latest_distance)

        horizontal_camera_angle = 62.2
        verticle_camera_angle = 48.8
        image_width = 2*self.current_distance*np.tan(horizontal_camera_angle/2/180*np.pi)
        image_height = 2*self.current_distance*np.tan(verticle_camera_angle/2/180*np.pi)

        old_distance = self.current_distance/zoom
        y_vel = -(self.current_distance - old_distance)#greater distance means further back, hence sign change

        self.prev_points = np.copy(cur_points)
        self.prev_descriptors = np.copy(cur_descriptors)

        message = Twist()
        message.linear.x = translation[0]/self.image.shape[1]*image_width
        message.linear.y = y_vel
        message.linear.z = translation[1]/self.image.shape[0]*image_height
        self.vel_pub.publish(message)

        ## docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        # next_dots, status, error = cv.calcOpticalFlowPyrLK(prev_image, gray_image, prev_dots, None, **lk_params)
        # self.prev_image = gray_image
        # self.prev_dots = next_dots

        #image = PIL_Image.open(BytesIO(data.data))

if __name__ == "__main__":
    features = OpticalFlow()
    features.start()