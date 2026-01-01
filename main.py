import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
from image_utils import load_image, edge_detection

# 1. טעינת תמונת צבע [cite: 21]
image_path = "street.jpg"  # וודא שהקובץ קיים בנתיב זה
img = load_image(image_path)

# 2. סינון רעשים בעזרת פילטר חציוני ball(3) [cite: 24, 25, 26]
# הערה: מאחר ו-ball הוא תלת-מימדי, נשתמש בשכבה אחת שלו עבור התמונה הדו-מימדית
footprint = ball(3)[3] 
clean_image = np.zeros_like(img)
for c in range(3):
    clean_image[:, :, c] = median(img[:, :, c], footprint)

# 3. הרצת פונקציית זיהוי הקצוות [cite: 28]
edges = edge_detection(clean_image)

# 4. הפיכת המערך לבינארי על-ידי בחירת ערך סף [cite: 28]
# נשתמש בערך סף 50 כפי שהגדרת, או לפי ניתוח ההיסטוגרמה [cite: 29, 30]
threshold = 50
edge_binary = edges > threshold

# 5. הדפסת התמונה ושמירתה כקובץ PNG 
plt.imshow(edge_binary, cmap="gray")
plt.title("Edge Detection Result")
plt.axis("off")
plt.show()

# שמירה לקובץ
result_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
result_image.save("my_edges.png")
print("התמונה נשמרה בהצלחה בשם my_edges.png")
