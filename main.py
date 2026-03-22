from matplotlib import pyplot as plt
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
======================================
            PREPROCESADO
======================================
"""


array1 = np.where(np.array(frames1) < 30,215, 0)

background = array1[-1].astype(np.float32)
frame = array1[1].astype(np.float32)

substracted = frame - background

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].imshow(frame, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original Frame')
axes[0].axis('off')

axes[1].imshow(background, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Background')
axes[1].axis('off')

axes[2].imshow(substracted, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Subtracted')
axes[2].axis('off')

plt.tight_layout()
plt.show()

"""
======================================
               TRACKPY
======================================
"""

