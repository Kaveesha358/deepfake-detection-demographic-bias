import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def extract_eye_region(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
        
    # Convert BGR to RGB since MTCNN expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    
    if results:
        keypoints = results[0]['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calculate bounding box covering both eyes with a safety margin
        min_x = min(left_eye[0], right_eye[0]) - 30
        max_x = max(left_eye[0], right_eye[0]) + 30
        min_y = min(left_eye[1], right_eye[1]) - 25
        max_y = max(left_eye[1], right_eye[1]) + 25
        
        h_img, w_img, _ = img_rgb.shape
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(w_img, max_x), min(h_img, max_y)
        
        eye_crop = img_rgb[min_y:max_y, min_x:max_x]
        if eye_crop.size == 0:
            
            return cv2.resize(img_rgb, (224, 224))
            
       
        return cv2.resize(eye_crop, (224, 224))
        
    else:
        print(f"Warning: No face/eyes detected in {image_path}")
        return None
