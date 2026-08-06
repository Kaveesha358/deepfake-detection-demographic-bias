import cv2
import numpy as np

def apply_jpeg_compression(image, quality=60):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def process_single_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (224, 224))
    
    if np.random.rand() > 0.5:
        q = np.random.randint(40, 90)
        img = apply_jpeg_compression(img, quality=q)
        
    if np.random.rand() > 0.7:
        img = apply_gaussian_blur(img, kernel_size=(3, 3))
        
    img = img.astype('float32') / 255.0
    
    return img
