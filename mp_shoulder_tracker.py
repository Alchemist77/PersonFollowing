#!/usr/bin/env python3

import rospy
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import mediapipe as mp
import math
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import math
from geometry_msgs.msg import Twist, Vector3



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
bridge = CvBridge()

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 640
# CAMERA_TOPIC = "/camera/color/image_raw"

CAMERA_DEPTH_TOPIC = "/camera/depth/image_raw"
CAMERA_RGB_TOPIC = "/camera/color/image_raw"
CAMERA_RGB_INFO = "/camera/color/camera_info"
CAMERA_DEPTH_INFO = "/camera/depth/camera_info"
CLOSE_TO_TARGET_TOPIC = "~shoulder_close_to_target"


def get_depth_pixels(shoulder_points):
    """
    Input: all the keypoints of mediapipe for the shoulder, passed in as a list of x,y coordinates
    Output: the mean depth of these keypoints
    """
    distance = 0
    if len(shoulder_points) == 0:
        return 0, 0
    px = sum([el[0] for el in shoulder_points])/len(shoulder_points)
    py = sum([el[1] for el in shoulder_points])/len(shoulder_points)
    return int(px), int(py)

def get_position_wrt_camera(info, depth_at_shoulder, cX, cY, width_ratio, height_ratio):
    m_fx = info.K[0]
    m_fy = info.K[4]
    m_cx = info.K[2]
    m_cy = info.K[5]
    inv_fx = 1. / m_fx
    inv_fy = 1. / m_fy
    depth_at_shoulder = depth_at_shoulder / 1000
    point_x = (cX - m_cx * width_ratio) * depth_at_shoulder * inv_fx
    point_y = (cY - m_cy * height_ratio) * depth_at_shoulder * inv_fy
    point_z = math.sqrt(depth_at_shoulder ** 2 - point_y ** 2 - point_x ** 2)
    return point_x, point_y, point_z


def get_shoulder_pixels_position(shoulder, landmark, image_width, image_height):
    """
    Left and right shoulder parameters are in position 15 to 22 of landmark array.
    Odd numbers are left and pair numbers are right shoulder.
    The landmarks are, ordered, for: shoulder, pinky, index, thumb
    """
    shoulder_points = []
    landmars_subset = landmark[11:13]
    #print("landmars_subset",landmars_subset)
    if shoulder == "RIGHT":
        ## Get info for elements at pair index position
        shoulder_landmarks = landmars_subset[0]
    if shoulder == "LEFT":
        ## Get info for elements at odd index position
        shoulder_landmarks = landmars_subset[1]
    #print("shoulder_landmarks",shoulder_landmarks)
    #for el in shoulder_landmarks:
    pos = (int(shoulder_landmarks.x * image_height), int(shoulder_landmarks.y * image_width))
    #print("POS 0 : %d, POS 1 %d" %(pos[0],pos[1]))
    if pos[0] < image_height and pos[1] < image_width:
        shoulder_points.append(pos)
    return shoulder_points


class shoulderTracker:

    def __init__(self):
        rospy.init_node('shoulderTracker', anonymous=False)
        self.pose_pub = rospy.Publisher("/shoulder_pub", PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.max_distance = 2.0
        self.min_distance = 1.2
        self.max_angle = 4.0
        self.cmd_vel = Twist()



        self.tss = ApproximateTimeSynchronizer(
            [Subscriber(CAMERA_RGB_TOPIC, Image, queue_size=1),
             Subscriber(CAMERA_RGB_INFO, CameraInfo, queue_size=1),
             Subscriber(CAMERA_DEPTH_TOPIC, Image, queue_size=1),
             Subscriber(CAMERA_DEPTH_INFO, CameraInfo, queue_size=1, buff_size=2 ** 24)],
            10,
            0.1,
            allow_headerless=True
        )
        self.tss.registerCallback(self.skeleton_position_callback)
        rospy.spin()

    def skeleton_position_callback(self, rgb_raw, rgb_info, depth_raw, depth_info):
        """
        Read the message od 2D skeleton position, then acquire the shoulder pixels
        and use the rgb_to_xyz method offered by ROS4HRI to get 3D position
        """
        try:
            image_r = bridge.imgmsg_to_cv2(rgb_raw, "bgr8")
            rgb = cv2.resize(image_r, (DESIRED_WIDTH, DESIRED_HEIGHT))
            # rgb = bridge.imgmsg_to_cv2(rgb_raw, "bgr8")
            image_d = bridge.imgmsg_to_cv2(depth_raw, "16UC1")
            depth = cv2.resize(image_d, (DESIRED_WIDTH, DESIRED_HEIGHT))
            # depth = bridge.imgmsg_to_cv2(depth_raw, "16UC1")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("rgb", rgb)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(10)
        # cv2.waitKey(10)

        # print(rgb.shape)
        # print(depth.shape)

        image_height, image_width, _ = rgb.shape
        old_height, old_width, _ = image_r.shape

        with mp_pose.Pose(static_image_mode=True,
                          min_detection_confidence=0.5,
                          model_complexity=1) as pose:
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                left_shoulder_pixels = get_shoulder_pixels_position("LEFT", results.pose_landmarks.landmark, image_height, image_width)
                right_shoulder_pixels = get_shoulder_pixels_position("RIGHT", results.pose_landmarks.landmark, image_height, image_width)

                left_depth_x, left_depth_y = get_depth_pixels(left_shoulder_pixels)
                right_depth_x, right_depth_y = get_depth_pixels(right_shoulder_pixels)

                print("X = %f, Y = %f" %(left_depth_x,left_depth_y))
                # print("X = %f, Y = %f" %(right_depth_x,right_depth_y))
                # print("depth size", depth.shape) # 480 x 640

                left_depth = depth[left_depth_y, left_depth_x]
                right_depth = depth[right_depth_y, right_depth_x]

                #print("left_depth",left_depth, "right_depth",right_depth)


                if right_depth == 0 or left_depth == 0 or left_depth > 2000 or right_depth > 2000:
                    return

                width_ratio = image_width / old_width
                height_ratio = image_height / old_height
                left_x, left_y, left_z = get_position_wrt_camera(depth_info, left_depth, left_depth_x, left_depth_y, width_ratio, height_ratio)
                right_x, right_y, right_z = get_position_wrt_camera(depth_info, right_depth, right_depth_x, right_depth_y, width_ratio, height_ratio)
                center_x, center_y, center_z = (left_x + right_x)/2, (left_y + right_y)/2, (left_z + right_z)/2


                # print("left_x, left_y, left_z", left_x, left_y, left_z)
                # print("right_x, right_y, right_z",right_x, right_y, right_z)
                #print("center x,y,z", center_x, center_y, center_z)
                yaw = math.atan2(center_x, center_z)

                #print("yaw_",yaw)


                # Print the shoulder angle
                # print("Shoulder angle: {:.2f} degrees".format(np.rad2deg(yaw)))

                distance_from_robot = math.sqrt((center_z)**2 + (center_x)**2)
                print(distance_from_robot)
                if distance_from_robot < self.min_distance:
                    self.cmd_vel.linear.x = 0
                else:
                    linear_speed = 0.2 * distance_from_robot
                    self.cmd_vel.linear.x = linear_speed
                    #print("distance_from_robot",distance_from_robot, " abs(np.rad2deg(yaw)", abs(np.rad2deg(yaw)) )
                if abs(np.rad2deg(yaw)) < self.max_angle:
                    self.cmd_vel.angular.z = 0
                else:
                    angular_speed = 0.02 * (self.max_angle - np.rad2deg(yaw))
                    self.cmd_vel.angular.z = angular_speed
                   # print("angular_speed",angular_speed)
                
                #self.cmd_vel.linear.x = 0
                print(self.cmd_vel)
                #if abs(self.cmd_vel.linear.x) < 0.1:
                #    self.cmd_vel.linear.x = 0
                self.cmd_vel_pub.publish(self.cmd_vel)

                pose = PoseStamped()

                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "camera_link"
                pose.pose.position.x = center_x
                pose.pose.position.y = center_y
                pose.pose.position.z = center_z

                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0

                self.pose_pub.publish(pose)

                 #print(right_x, right_y, right_z)
                # print(right_shoulder_pixels)


                # Draw pose landmarks.
                 #print(f'Pose landmarks of Skeleton:')
                # annotated_image = rgb.copy()
                # mp_drawing.draw_landmarks(
                #     annotated_image,
                #     results.pose_landmarks,
                #     mp_pose.POSE_CONNECTIONS,
                #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # cv2.imshow("Windows", annotated_image)
                # cv2.waitKey(1)


if __name__ == "__main__":
    try:
        shoulderTracker()
    except (rospy.ROSInterruptException, Exception, KeyboardInterrupt) as e:
        raise e
