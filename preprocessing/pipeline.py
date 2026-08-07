!pip install mtcnn

import cv2
import numpy as np
import os
from mtcnn import MTCNN

detector = MTCNN()

#Face Detection and Cropping
def detect_and_crop_face(image):
    result = detector.detect_faces(image)
    if result:
        x, y, w, h = result[0]['box']
        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]
        return cv2.resize(face, (224, 224))
    return None

# Compression and Blur functions
def apply_jpeg_compression(image, quality=60):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

#Full pipeline
def process_single_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = detect_and_crop_face(img_rgb)
    if face is None:
        return None 

    img = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    if np.random.rand() > 0.5:
        q = np.random.randint(40, 90)
        img = apply_jpeg_compression(img, quality=q)

    if np.random.rand() > 0.7:
        img = apply_gaussian_blur(img, kernel_size=(3, 3))

    img = img.astype('float32') / 255.0

    return img

print("Full preprocessing pipeline ready (face detection + compression + blur)")
