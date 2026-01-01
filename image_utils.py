from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינת התמונה והפיכתה למערך נומפי [cite: 2]
    image = Image.open(path).convert("RGB")
    return np.array(image)

def edge_detection(image):
    # הפיכת המערך לתמונה אפורה על-ידי מיצוע שלושת ערוצי הצבע [cite: 6, 7]
    gray = np.mean(image, axis=2).astype(np.float64)
    
    # בניית פילטר לשינויים בכיוון האנכי (kernelY) [cite: 8, 10]
    kernelY = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]])
    
    # בניית פילטר לשינויים בכיוון האופקי (kernelX) [cite: 9, 10]
    kernelX = np.array([[1, 0, -1], 
                        [2, 0, -2], 
                        [1, 0, -1]])
    
    # הפעלת הקונבולוציה עם padding=0 ושמירה על גודל תמונה מקורי [cite: 12, 13]
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    
    # חישוב עוצמת הקצוות לפי הנוסחה [cite: 18, 19]
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
