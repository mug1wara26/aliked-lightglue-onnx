import numpy as np
import onnx
import onnxruntime as rt
import cv2 as cv
from einops import repeat, rearrange
from copy import deepcopy


def plot_keypoints(image, kpts, radius=2, color=(0, 0, 255)):
    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        x0, y0 = kpt
        cv.circle(out, (x0, y0), radius, color, -1, lineType=cv.LINE_4)
    return out


model = onnx.load("models/aliked-n16rot-top1k-tum.onnx")
onnx.checker.check_model(model)

sess = rt.InferenceSession("models/aliked-n16rot-top1k-tum.onnx")

image = cv.imread("assets/1.png")

image = image.astype("float32")

image = repeat(image, 'h w c -> n h w c', n = 1)
image = rearrange(image, 'n h w c -> n c h w')

keypoints = np.array(sess.run(['keypoints'], {'image': image}))
print(keypoints)
keypoints = (np.array([639, 479]) * (keypoints + 1) / 2)[0]

vis_img = plot_keypoints(cv.resize(cv.imread("assets/1.png"), (640,480)), keypoints)

cv.imshow('image', vis_img)
cv.waitKey(0)
