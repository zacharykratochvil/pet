#!/usr/bin/python
import cv2
#from PIL import Image as PIL_Image
#from io import BytesIO
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

class VisionFeatures:

    def __init__(self):
        self.camera_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.on_image, queue_size=1)
        
        self.prev_image = None
        self.prev_dots = []

        #self.particle_pub = rospy.Publisher("feature_cloud", PointCloud, queue_size=1)
        rospy.init_node("vision_features",anonymous=False)

    def start(self):
        rospy.spin()

    def on_image(self, data):        

        '''
        bytes_per_pixel = int(data.step/data.width)
        channels = 3
        byte_depth = int(bytes_per_pixel/channels)
        rospy.loginfo(f"bytes: {bytes_per_pixel}, depth: {byte_depth}")

        image = np.empty((data.height, data.width, 3))
        byte_i = 0
        for x in range(data.width):
            for y in range(data.height):

                def get_value(start, end):
                    _bytes = data.data[start:end]
                    return int.from_bytes(_bytes, byteorder='big', signed=False)

                r = get_value(byte_i,               byte_i+byte_depth)
                g = get_value(byte_i+byte_depth,    byte_i+2*byte_depth)
                b = get_value(byte_i+2*byte_depth,  byte_i+3*byte_depth)

                image[y,x,:] = (b,g,r)

                byte_i += bytes_per_pixel
        '''

        #int_data = int.from_bytes(data.data,byteorder='little',signed=False)
        int_data = np.frombuffer(data.data, np.uint8)
        image = cv2.imdecode(int_data, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_image = None:
            self.prev_image = gray_image

        # docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        next_dots, status, error = cv.calcOpticalFlowPyrLK(prev_image, gray_image, prev_dots, None, **lk_params)
        self.prev_image = gray_image
        self.prev_dots = next_dots

        #image = PIL_Image.open(BytesIO(data.data))
        cv2.imshow("image", image)
        cv2.waitKey()

if __name__ == "__main__":
    features = VisionFeatures()
    features.start()