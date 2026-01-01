from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינת התמונה ללא כפיית RGB כדי לאפשר השוואה תקינה בטסטים [cite: 2]
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    # 1. הפיכת המערך לתמונה אפורה בעלת ערוץ אחד על-ידי מיצוע [cite: 6, 7]
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float64)
    else:
        gray = image.astype(np.float64)
    
    # 2. בניית פילטרים לפי הערכים המדויקים בנספח 
    # פילטר לשינויים באנכי (שורות)
    kernelY = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]])
    
    # פילטר לשינויים באופקי (עמודות)
    kernelX = np.array([[1, 0, -1], 
                        [2, 0, -2], 
                        [1, 0, -1]])
    
    # 3. הפעלת קונבולוציה בעזרת scipy.signal.convolve2d [cite: 12, 15]
    # שימוש ב-padding=0 (fillvalue=0) ושמירה על גודל מקורי (mode='same') [cite: 13]
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    
    # 4. חישוב עוצמת הקצוות לפי הנוסחה [cite: 18, 19]
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
