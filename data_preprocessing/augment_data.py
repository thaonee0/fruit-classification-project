import cv2
import numpy as np
import random

def rotate_image(image, angle):
    """Xoay ảnh với góc chỉ định."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def flip_image(image, flip_code):
    """Lật ảnh (flip_code=0: lật dọc, flip_code=1: lật ngang)."""
    return cv2.flip(image, flip_code)

def augment_image(image):
    """Thực hiện augment bằng cách xoay và lật ảnh."""
    augmented_images = []
    
    # Xoay ảnh ở 3 góc: 90, 180, 270 độ
    for angle in [90, 180, 270]:
        augmented_images.append(rotate_image(image, angle))
    
    # Lật ảnh ngang và dọc
    augmented_images.append(flip_image(image, 0))  # Lật dọc
    augmented_images.append(flip_image(image, 1))  # Lật ngang
    
    return augmented_images
