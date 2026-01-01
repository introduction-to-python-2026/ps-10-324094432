import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
from image_utils import load_image, edge_detection

# Load image
image_path = "/content/street.jpg"  # שנה לפי הנתיב של התמונה שלך
img = load_image(image_path)

# Apply median filter directly to RGB image using ball(3)
clean_image = median(img, ball(3))

# Edge detection
edges = edge_detection(clean_image)

# Binary threshold (fixed)
edge_binary = edges > 50

# Display
plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()

# Save as PNG
edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save("my_edges.png")
