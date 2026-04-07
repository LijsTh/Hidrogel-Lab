from matplotlib import pyplot as plt
import matplotlib.patches as patches
import trackpy as tp
from skimage import io 
import pandas as pd
import pims
import numpy as np

# .tif 
PATH_2FPS = "./../info/Silo/2fps.tif"
PATH_10FPS = "./../info/Silo/10fps.tif"

frames1 = pims.open(PATH_2FPS)
frames2 = pims.open(PATH_10FPS)

"""
================================================
                PREPROCESADO
================================================
"""


array1 = np.where(np.array(frames1) < 30,215, 0)

background = array1[-1].astype(np.float32)
frame = array1[1].astype(np.float32)

substracted = frame - background

fig, axes = plt.subplots(1, 4, figsize=(10, 5))
axes[0].imshow(frame, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original Frame')
axes[0].axis('off')

axes[1].imshow(background, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Background')
axes[1].axis('off')

axes[2].imshow(substracted, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Subtracted')
axes[2].axis('off')

features = tp.locate(substracted, 7)
axes[3].imshow(substracted, cmap="gray", vmin=0, vmax=255)
axes[3].scatter(
    features["x"],
    features["y"],
    s=40,
    facecolors="none",
    edgecolors="red",
    linewidths=1,
)
axes[3].set_title("Detected particles")
axes[3].axis("off")

plt.tight_layout()
plt.show()

"""
=================================================
            REGIONS OF INTEREST EXTRACTION
=================================================
"""

"""
- Ancho superior del silo, 𝑊 = 345 𝑚𝑚,
- Largo del silo, 𝐻 = 700 𝑚𝑚
"""

# TODO: CHECKEAR ESTO MEJOR
PIXELS_PER_MM = 1.4 # PONELE QUE ESTO LO HICE CON LA REGLA EN FIJI, SOS LIBRE JOACO DE VERIFICAR 

# The y-coordinate (in pixels) of the bottom reference point
BOTTOM_REF_Y_PX = substracted.shape[0] 

# 1. Setup the figure
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(substracted, cmap='gray', vmin=0, vmax=255)
ax.set_title('Verifying ROI Cuts')

# 2. Define a function to draw the boxes
def draw_roi_box(ax, bottom_offset_mm, height_mm, px_per_mm, y_ref_px, width_px, color='red'):
    """Calculates the coordinates and draws a rectangle on the given axes."""
    # Convert mm to pixels
    bottom_offset_px = int(bottom_offset_mm * px_per_mm)
    height_px = int(height_mm * px_per_mm)
    
    # Calculate the Y coordinates (y=0 is the top)
    y_bottom_idx = y_ref_px - bottom_offset_px
    y_top_idx = y_bottom_idx - height_px
    
    # The (x,y) parameter for matplotlib Rectangle is the top-left corner
    top_left_corner = (0, y_top_idx) 
    
    # Create the rectangle patch
    rect = patches.Rectangle(
        top_left_corner, 
        width=width_px, 
        height=height_px, 
        linewidth=2, 
        edgecolor=color, 
        facecolor='none'
    )
    
    # Add the rectangle to the plot
    ax.add_patch(rect)
    
    # Add a text label right above the box so you know which is which
    label = f"{bottom_offset_mm}mm - {bottom_offset_mm + height_mm}mm"
    ax.text(10, y_top_idx - 10, label, color=color, fontsize=10, weight='bold')

# 3. Draw the three boxes
image_width = substracted.shape[1]

draw_roi_box(ax, 10, 30, PIXELS_PER_MM, BOTTOM_REF_Y_PX, image_width)
draw_roi_box(ax, 160, 30, PIXELS_PER_MM, BOTTOM_REF_Y_PX, image_width)
draw_roi_box(ax, 460, 30, PIXELS_PER_MM, BOTTOM_REF_Y_PX, image_width)

plt.show()
"""
================================================
                    TRACKPY
================================================
"""

