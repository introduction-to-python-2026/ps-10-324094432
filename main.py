import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from PIL import Image
from image_utils import load_image, edge_detection

# 1. Load the color image
image_path = "/content/street.jpg"  # שנה לפי הנתיב של התמונה שלך
img = load_image(image_path)

# 2. Suppress noise with median filter (applied per channel)
clean_image = np.zeros_like(img)
for c in range(3):
    clean_image[:, :, c] = median(img[:, :, c], disk(3))  # disk(3) footprint 2D

# 3. Run edge detection
edges = edge_detection(clean_image)

# 4. Convert edgeMAG to binary using a threshold
threshold = 0.25 * edges.max()  # אפשר לשנות את הערך לפי הצורך
edge_binary = edges > threshold

# 5. Display the binary edge image
plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()

# 6. Save binary edge image as PNG
edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save("my_edges.png")
