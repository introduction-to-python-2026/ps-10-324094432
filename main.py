import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
from image_utils import load_image, edge_detection

# 1. טעינת תמונת הצבע ששלחת [cite: 21]
img = load_image("street.jpg")

# 2. סינון רעשים בעזרת פילטר חציוני ball(3) [cite: 24, 26]
# הפונקציה median תטפל בכל ערוץ בנפרד באופן אוטומטי
clean_image = median(img, ball(3))

# 3. הרצת פונקציית זיהוי הקצוות [cite: 28]
edges = edge_detection(clean_image)

# 4. הפיכת המערך לבינארי על-ידי בחירת ערך סף (למשל 50) [cite: 28]
threshold = 50
edge_binary = edges > threshold

# 5. הצגת התמונה ושמירתה כקובץ PNG [cite: 30]
plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()

# שמירה לקובץ כפי שנדרש
result_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
result_image.save("my_edges.png")
print("Saved edge detection result to my_edges.png")
