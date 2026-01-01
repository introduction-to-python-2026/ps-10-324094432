import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk  # use disk, not ball
from PIL import Image
from image_utils import load_image, edge_detection

# Load image
image_path = "/content/street.jpg"
img = load_image(image_path)

# Apply median filter per channel using disk(3)
clean_image = np.zeros_like(img)
for c in range(3):
    clean_image[:, :, c] = median(img[:, :, c], disk(3))  # disk(3) footprint 2D

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
