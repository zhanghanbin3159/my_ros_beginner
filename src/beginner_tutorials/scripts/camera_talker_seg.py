#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def talker():
     pub = rospy.Publisher('/tutorial/image', Image, queue_size=1) 
     rospy.init_node('talker', anonymous=True) 
     rate = rospy.Rate(30)
     bridge = CvBridge()
     Video = cv2.VideoCapture(0)
     while not rospy.is_shutdown():
         ret, img = Video.read()
         cv2.imshow("talker", img)
         cv2.waitKey(3)
         pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
         rate.sleep()

if __name__ == '__main__':
     try:
         talker()
     except rospy.ROSInterruptException:
         pass
