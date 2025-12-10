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

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
import joblib

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
            
        # Determine type of saved model (keras or joblib)
        self.model_type =None
        try:
            splt = self.classification_model_name.split('.')
            if len(splt)!=2:
                raise ValueError(f'Error determining model type: {self.classification_model_name}')
            self.model_type = splt[-1]
            if self.model_type not in ['keras', 'joblib']:
                raise ValueError(f'Classification model type not supported: {tp}')
        except Exception as e:
            self.get_logger().error(e)
            self.classification_model = None
            exit(1)
            
        # Try to load the hand gesture keras models
        try:
            if self.model_type == 'keras':
                self.classification_model = keras.models.load_model(self.classification_model_path)
            elif self.model_type == 'joblib':
                self.classification_model = joblib.load(self.classification_model_path)
            else:
                raise ValueError('Model type is invalid')
            # self.classification_model.trainable = False
            self.get_logger().info(f'Loaded hand gesture model: {self.classification_model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load hand gesture model: {e}')
            self.classification_model = None
            exit(1)
            
        self.predict_fun = {
            'joblib': self.classification_model.predict_proba,
            'keras': self.classification_model.predict
        }[self.model_type]
                    
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
    
    def publish_failure(self, message =  None):
        if message is not None:
            self.get_logger().error(message)
        out = GestureMsg()
        out.detected = False
        self.publisher.publish(out)
        return
    
    def show_timing_info(self, elapsed_time_1, elapsed_time_2, elapsed_time_3):
        self.get_logger().info(f'Inference time')
        self.get_logger().info(f'\tMP: {elapsed_time_1:.4f} s') 
        self.get_logger().info(f'\tPPN: {elapsed_time_2:.4f} s') 
        self.get_logger().info(f'\tLD: {elapsed_time_3:.4f} s') 
        self.get_logger().info(f'\tT: {elapsed_time_1 + elapsed_time_2 + elapsed_time_3:.4f} s') 
       
    def acq_callback(self, msg: AcquisitionMsg):
        start_time = time.time()       
        image_msg = msg.image
        
        try:
            # convert ROS image to OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            return self.publish_failure(f'Failed to convert ROS Image to CV2: {e}')
        if self.hand_landmarker is None:
            return self.publish_failure('Hand landmarker not initialized, skipping inference')
        if self.classification_model is None:
            return self.publish_failure('Classification model not loaded')
        
        # MediaPipe expects RGB data for ImageFormat.SRGB
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # preferred path when mediapipe.tasks.python.vision.core.Image is available
        mp_image = MPImage(image_format=MPImageFormat.SRGB, data=rgb)
        results = self.hand_landmarker.detect(mp_image)       
        # Log the detection results
        if results.hand_landmarks:
            pass
        else:
            return self.publish_failure()
            
        hand_landmarks = results.hand_landmarks[0]
        
        if len(hand_landmarks)==0:
            return self.publish_failure()
        
        # Define message
        out_msg = GestureMsg()
        out_msg.detected = True
            
        # De-normalize landmark coordinates
        h, w = cv_image.shape[:2]
        
        # Transform landmarks to (21, 2) numpy array
        landmarks_array = np.array([[l.x * w, l.y * h] for l in hand_landmarks], dtype=np.float32)
                   
        # Bounding Box estimation
        min_x, min_y = np.min(landmarks_array, axis=0)
        max_x, max_y = np.max(landmarks_array, axis=0)
        
        # Force rectangle for CNN
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        d = 1.1 * max((max_y - min_y), (max_x - min_x))
        
        out_msg.x_max = max(0, int(cx + 0.5 * d))
        out_msg.x_min = max(0, int(cx - 0.5 * d))
        out_msg.y_max = min(w, int(cy + 0.5 * d))
        out_msg.y_min = min(h, int(cy - 0.5 * d))
                
        elapsed_time_1 = time.time() - start_time
        
        # Preprocess landmarks for classification
        processed_landmarks = self.preprocess_norm(landmarks_array)
        
        out_msg.landmarks_x, out_msg.landmarks_y = landmarks_array.T.astype(int).tolist()

        # Predict gesture using classification model
        elapsed_time_2 = time.time() - start_time - elapsed_time_1
        # Reshape to batch format (1, 42) for model input
        model_input = processed_landmarks.reshape(1, -1)
        
        # Predict Class
        prediction = self.predict_fun(model_input)
        
        # Translate to predicted_class and confidence
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        # Determine Unknown class
        if confidence < self.min_confidence or predicted_class == 3:
            predicted_class = 3
        
        out_msg.player_gesture = predicted_class
        out_msg.confidence = confidence
        
        out_msg.inference_time = time.time() - start_time
                    
        self.publisher.publish(out_msg)
        
        elapsed_time_3 = time.time() - start_time - elapsed_time_1 - elapsed_time_2
        # self.show_timing_info(elapsed_time_1, elapsed_time_2, elapsed_time_3)
        
                

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