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


frames = np.where(np.array(frames1) < 40, 215, 0)

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

## TRACKPY CONIFG
tp.quiet()
features = tp.batch(frames_substracted, diameter=9, minmass=500)
t = tp.link(features, search_range=5, memory=3)
t = tp.filter_stubs(t, 10)

particles_delta = t.groupby("particle").aggregate({"y": ["max", "min"]})
particles_with_enough_y_movement = particles_delta[
    particles_delta[("y", "max")] - particles_delta[("y", "min")] > 30
]
tp.plot_traj(t)
t = t[t.particle.isin(particles_with_enough_y_movement.index)]

t.reset_index(drop=True, inplace=True)
velocities_per_trayectory = []
print(t.particle.nunique())
for p in t.particle.unique():
    p_trayectory = t[t.particle == p].sort_values(by="frame")
    p_trayectory["velocity_y"] = p_trayectory.y.shift(1) - p_trayectory.y
    p_trayectory["velocity_x"] = p_trayectory.x.shift(1) - p_trayectory.x
    velocities_per_trayectory.append(
        p_trayectory[["x", "y", "velocity_y", "velocity_x", "frame"]]
    )



frames = t.frame.unique()
velocities = pd.concat(velocities_per_trayectory)
print(velocities.columns)
for f in frames:
    velocities[velocities.frame == f].to_csv(f"data/velocities_{f}.csv", index=False)
