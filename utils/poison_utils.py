import cv2
import numpy as np
from PIL import Image

_KERNEL_5x5 = np.ones((5, 5))

def erode_image(img, iters):
    if isinstance(img, Image.Image):
        arr = np.array(img.convert('L'))
        input_is_pil = True
    else:
        arr = np.asarray(img).copy()
        input_is_pil = False

    eroded = cv2.erode(arr, _KERNEL_5x5, iterations=int(iters))

    if input_is_pil:
        return Image.fromarray(eroded)
    return eroded
