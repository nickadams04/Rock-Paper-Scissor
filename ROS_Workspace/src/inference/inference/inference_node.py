import os
import time
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from custom_msgs.msg import AcquisitionMsg, GestureMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options


from mediapipe import Image as MPImage
from mediapipe import ImageFormat as MPImageFormat

import tensorflow as tf
from tensorflow import keras
import numpy as np


class InferenceNode(Node):
    def parameters(self):
        self.declare_parameter('detection_model', 'hand_landmarker.task')
        self.declare_parameter('classification_model', 'hand_gesture_model.keras')
        self.declare_parameter('min_confidence', 0.6)
        
        self.detection_model_name = self.get_parameter('detection_model').value
        self.classification_model_name = self.get_parameter('classification_model').value
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        
        
    def __init__(self):
        super().__init__('inference_node')
        self.parameters()
       
        share_dir = get_package_share_directory('inference')
        models_path = os.path.join(share_dir, '../../../..', 'src/inference/models/')
        self.detection_model_path = os.path.join(models_path, self.detection_model_name)
        self.classification_model_path = os.path.join(models_path, self.classification_model_name)
        self.get_logger().info(os.path.abspath(self.classification_model_path))
        
        # bridge for ROS Image <-> OpenCV conversions
        self.bridge = CvBridge()
        
        # try to load the MediaPipe hand landmarker model
        try:
            base_options = mp_base_options.BaseOptions(model_asset_path=self.detection_model_path)
            opts = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                running_mode=vision.RunningMode.IMAGE
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(opts)
            self.get_logger().info(f'Loaded hand landmarker model: {self.detection_model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load hand landmarker model: {e}')
            self.hand_landmarker = None
            exit(1)
            
        # Try to load the hand gesture keras models
        try:
            self.classification_model = keras.models.load_model(self.classification_model_path)
            self.get_logger().info(f'Loaded hand gesture model: {self.classification_model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load hand gesture model: {e}')
            self.classification_model = None
            exit(1)
                    
        # subscription
        self.subscriber = self.create_subscription(AcquisitionMsg, '/image_raw', self.acq_callback, 10)
        self.publisher = self.create_publisher(GestureMsg, '/gesture', 10)
        
    def preprocess_norm(self, landmarks):
        landmarks = np.array(landmarks, dtype=np.float32)

        # Rebase on wrist
        p0 = landmarks[0]
        landmarks = landmarks - p0

        # determine hand x axis
        vx, vy = (landmarks[17][0] - landmarks[5][0], landmarks[17][1] - landmarks[5][1])
        # guard degenerate axis
        eps = 1e-6
        if np.isfinite(vx) and np.isfinite(vy) and (abs(vx) > eps or abs(vy) > eps):
            theta = -np.arctan2(vy, vx)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s],[s,  c]], dtype=np.float32)
            landmarks = landmarks @ R.T

        # ensure palm faces upwards
        fingertip_indices = [8, 12, 16, 20]
        mean_y = np.mean(landmarks[fingertip_indices, 1])
        if np.isfinite(mean_y) and mean_y < 0:
            landmarks[:, 1] *= -1

        # normalize lengths (avoid divide by zero)
        scale = np.linalg.norm(landmarks[17] - landmarks[5]) + eps
        landmarks = landmarks / scale

        # sanitize
        landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)

        return landmarks.reshape(-1).astype(np.float32)
       
    def acq_callback(self, msg: AcquisitionMsg):
        # basic logging
        # self.get_logger().info(f'Got frame {getattr(msg, "global_idx", "unknown")}')        
        image_msg = msg.image
        try:
            # convert ROS image to OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS Image to CV2: {e}')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return
        
        if self.hand_landmarker is None:
            self.get_logger().warning('Hand landmarker not initialized, skipping inference')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return
        
        # MediaPipe expects RGB data for ImageFormat.SRGB
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # preferred path when mediapipe.tasks.python.vision.core.Image is available
        start_time = time.time()
        mp_image = MPImage(image_format=MPImageFormat.SRGB, data=rgb)
        results = self.hand_landmarker.detect(mp_image)
        elapsed_time = time.time() - start_time
        # self.get_logger().info(f'Inference time: {elapsed_time:.4f} seconds')        
        # Log the detection results
        if results.hand_landmarks:
            pass
            # self.get_logger().info(f'Detected {len(results.hand_landmarks)} hand(s)')
            # for idx, hand_landmarks in enumerate(results.hand_landmarks):
            #     self.get_logger().info(f'Hand {idx}: {len(hand_landmarks)} landmarks detected')
            #     self.get_logger().info(f'L0: {hand_landmarks[0].x}')
        else:
            # self.get_logger().info('No hands detected')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return 
            
        hand_landmarks = results.hand_landmarks[0]
        
        if len(hand_landmarks)==0:
            # self.get_logger().info('No keypoints detected')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return 
            
        # De-normalize landmark coordinates
        h, w = cv_image.shape[:2]
        for landmark in hand_landmarks:
            landmark.x = int(landmark.x * w)
            landmark.y = int(landmark.y * h) 
                   
        # Bounding Box estimation
        max_x = max([l.x for l in hand_landmarks])
        max_y = max([l.y for l in hand_landmarks])
        min_x = min([l.x for l in hand_landmarks])
        min_y = min([l.y for l in hand_landmarks])
        
        # Force rectangle for CNN
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        d = 1.1 * max((max_y - min_y), (max_x - min_x))
        
        bbox_x_max = cx + 0.5 * d
        bbox_x_min = cx - 0.5 * d
        bbox_y_max = cy + 0.5 * d
        bbox_y_min = cy - 0.5 * d
        
        # Crop image to bbox
        # Ensure bbox is within image bounds
        bbox_x_min = max(0, int(bbox_x_min))
        bbox_y_min = max(0, int(bbox_y_min))
        bbox_x_max = min(w, int(bbox_x_max))
        bbox_y_max = min(h, int(bbox_y_max))

        # # Crop the image
        # cropped_image = cv_image[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max]
        
        # # Resize cropped image to 224x224
        # IMG_SIZE = (224, 224)
        # cropped_image = cv2.resize(cropped_image, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Transform landmarks to (21, 2) numpy array
        landmarks_array = np.array([[l.x, l.y] for l in hand_landmarks], dtype=np.float32)

        # Preprocess landmarks for classification
        processed_landmarks = self.preprocess_norm(landmarks_array)
        
        out_msg = GestureMsg()
        out_msg.landmarks_x = [int(l.x) for l in hand_landmarks]
        out_msg.landmarks_y = [int(l.y) for l in hand_landmarks]
        
        out_msg.x_max = bbox_x_max
        out_msg.x_min = bbox_x_min
        out_msg.y_max = bbox_y_max
        out_msg.y_min = bbox_y_min
        
        out_msg.detected = True

        # Predict gesture using classification model
        if self.classification_model is not None:
            # Reshape to batch format (1, 42) for model input
            model_input = processed_landmarks.reshape(1, -1)
            prediction = self.classification_model.predict(model_input, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            threshold = 0.85
            if confidence < self.min_confidence:
                predicted_class = 3
            
            # Map class index to gesture name
            # gesture_names = ['rock', 'paper', 'scissors']
            # gesture = gesture_names[predicted_class] if predicted_class < len(gesture_names) else 'unknown'
            
            # self.get_logger().info(f'Predicted gesture: {gesture} (confidence: {confidence:.2f})')
            # self.get_logger().info(f'Sum is {np.sum(prediction[0])}')
            # out_msg.player_gesture = predicted_class if predicted_class in [0, 1, 2] else 255
            # ensure a native Python int (np.int64 would fail PyLong_Check in ROS2 conversion)
            pc = int(predicted_class)
            out_msg.player_gesture = pc
            out_msg.confidence = float(confidence)
        else:
            self.get_logger().warning('Classification model not loaded')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return 
            
        
        self.publisher.publish(out_msg)
        
                

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # graceful shutdown on Ctrl-C
        node.get_logger().info('Interrupted by user, shutting down.')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()