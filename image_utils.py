from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינת התמונה ללא כפיית RGB כדי לאפשר השוואה בטסטים
    image = Image.open(path)
    return np.array(image)

def edge_detection(image):
    # בדיקה: אם התמונה היא צבעונית (3 ערוצים), נבצע מיצוע
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float64) [cite: 7]
    else:
        gray = image.astype(np.float64)
    
    # הגדרת הפילטרים בדיוק לפי סעיף 2 בנספח 
    kernelY = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]])
    
    kernelX = np.array([[1, 0, -1], 
                        [2, 0, -2], 
                        [1, 0, -1]])
    
    # ביצוע קונבולוציה עם padding=0 (fillvalue=0) ושמירה על גודל מקורי [cite: 13]
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0) [cite: 12]
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0) [cite: 12]
    
    # חישוב עוצמת הקצוות לפי הנוסחה [cite: 19]
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
