import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
from image_utils import load_image, edge_detection

# טעינת התמונה
image_path = "street.jpg" 
img = load_image(image_path)

# ניקוי רעשים בעזרת פילטר חציוני ball(3) לפי ההוראות [cite: 26]
# skimage יודע לטפל ב-ball(3) על תמונת RGB אם מעבירים אותה ככה
clean_image = median(img, ball(3))

# זיהוי קצוות
edges = edge_detection(clean_image)

# יצירת תמונה בינארית לפי ערך סף 
threshold = 50
edge_binary = edges > threshold

# שמירה והצגה [cite: 30]
result_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
result_image.save("my_edges.png")

plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()
