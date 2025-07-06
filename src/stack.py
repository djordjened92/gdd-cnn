import os
import cv2
import numpy as np
from glob import glob

img_paths = glob('grad_cam_dir/*.png')
img_paths = sorted(img_paths)

imgs = []
for p in img_paths:
    img = cv2.imread(p)
    imgs.append(img)

stacked = np.hstack(imgs)
cv2.imwrite('grad_cam_stacked.png', stacked)