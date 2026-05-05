import numpy as np
import pandas as pd
import pims
import trackpy as tp
from matplotlib import pyplot as plt

# .tif
PATH_2FPS = "2fps.tif"
PATH_10FPS = "10fps.tif"

frames1 = pims.open(PATH_2FPS)
frames2 = pims.open(PATH_10FPS)

"""
================================================
                PREPROCESADO
================================================
"""


frames = np.where(np.array(frames1) < 30, 215, 0)

background = frames[-1].astype(np.float32)
frames = [f.astype(np.float32) for f in frames]
# frame_1 = frames[1].astype(np.float32)

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].imshow(frames[1], cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original Frame")
axes[0].axis("off")


axes[1].imshow(background, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Background")
axes[1].axis("off")

# substracted = frame_1 - background
frames_substracted = [f - background for f in frames]
frames_substracted = np.clip(frames_substracted, 0, 255).astype(np.uint8)
axes[2].imshow(frames_substracted[1], cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Subtracted")
axes[2].axis("off")

tp.quiet()
features = tp.batch(frames_substracted, 9, minmass=500)
# features = features[features.y < 5 * (features.x + 30)]
# features = features[features.y < 30]
# # Borde izquierdo
# img_h, img_w = frames_substracted[1].shape[:2]
# x_line = np.array([0, min(img_h - 1, (img_w - 1) // 2)])
# y_line = 5 * (x_line + 30) #0.15 un poquito cruzado
# axes[0].plot(x_line, y_line, color="red", linewidth=1.5)
# axes[1].plot(x_line, y_line, color="red", linewidth=1.5)
# axes[2].plot(x_line, y_line, color="red", linewidth=1.5)
# #Filter the features below this line
# features = features[features.y < 5 * (features.x + 30)]

# axes[2].axline((0,0), (1,1), color="blue", linewidth=1.5)
# #Filter the features below this line
# # features = features[features.y < 5 * (features.x + 30)]
# # print(features.describe())
t = tp.link(features, 5, memory=3)
t = tp.filter_stubs(t, 10)
# tp.plot_traj(t)

particles_delta = t.groupby("particle").aggregate({"y": ["max", "min"]})
particles_with_enough_y_movement = particles_delta[
    particles_delta[("y", "max")] - particles_delta[("y", "min")] > 30
]
t = t[t.particle.isin(particles_with_enough_y_movement.index)]
# tp.plot_traj(t)

t.reset_index(drop=True, inplace=True)
velocities_per_trayectory = []
for p in t.particle.unique():
    p_trayectory = t[t.particle == p].sort_values(by="frame")
    p_trayectory["velocity_y"] = p_trayectory.y.shift(1) - p_trayectory.y
    p_trayectory["velocity_x"] = p_trayectory.x.shift(1) - p_trayectory.x
    velocities_per_trayectory.append(
        p_trayectory[["x", "y", "velocity_y", "velocity_x", "frame"]]
    )


def get_velocity_field_at_frame(velocities_per_trayectory, frame):
    alto = 1024
    ancho = 472


frames = t.frame.unique()
velocities = pd.concat(velocities_per_trayectory)
print(velocities.columns)
for f in frames:
    velocities[velocities.frame == f].to_csv(f"data/velocities_{f}.csv", index=False)
# velocity_field_per_frame = [get_velocity_field_at_frame(velocities_per_trayectory, f) for f in tqdm(frames[100:101])]

'''
"""
=================================================
            REGIONm OF INTEREST EXTRACTION
=================================================
"""

"""
- Ancho superior del silo, 𝑊 = 345 𝑚𝑚,
- Largo del silo, 𝐻 = 700 𝑚𝑚
"""

# TODO: CHECKEAR ESTO MEJOR
PIXELS_PER_MM = (
    1.4  # PONELE QUE ESTO LO HICE CON LA REGLA EN FIJI, SOS LIBRE JOACO DE VERIFICAR
)

# The y-coordinate (in pixels) of the bottom reference point
BOTTOM_REF_Y_PX = frames_substracted[0].shape[0]

# 1. Setup the figure
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(frames_substracted[0], cmap="gray", vmin=0, vmax=255)
ax.set_title("Verifying ROI Cuts")


# 2. Define a function to draw the boxes
def draw_roi_box(
    ax, bottom_offset_mm, height_mm, px_per_mm, y_ref_px, width_px, color="red"
):
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
        facecolor="none",
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)

    # Add a text label right above the box so you know which is which
    label = f"{bottom_offset_mm}mm - {bottom_offset_mm + height_mm}mm"
    ax.text(10, y_top_idx - 10, label, color=color, fontsize=10, weight="bold")


# 3. Draw the three boxes
image_width = frames_substracted[0].shape[1]

draw_roi_box(ax, 10, 30, PIXELS_PER_MM, BOTTOM_REF_Y_PX, image_width)
draw_roi_box(ax, 160, 30, PIXELS_PER_MM, BOTTOM_REF_Y_PX, image_width)
draw_roi_box(ax, 460, 30, PIXELS_PER_MM, BOTTOM_REF_Y_PX, image_width)

plt.show()
"""
================================================
                    TRACKPY
================================================
"""
'''
