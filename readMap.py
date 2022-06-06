import numpy as np
import cv2
 

obstacles = np.empty((0, 2), dtype=int)
print(obstacles)
img = cv2.imread('map_2.png', cv2.IMREAD_COLOR)
new_img = np.zeros((100, 100, 3))
for i in range(100):
    for j in range(100):
        if sum(img[i, j]) > 384:
            obstacles = np.append(obstacles, [[i, j]], axis=0)
            new_img[i, j] = [0, 0, 0]
        else:
            new_img[i, j] = [255, 255, 255]
print(obstacles)
print(new_img[76, 23], new_img[14, 58])
new_img[76, 23] = [0, 0, 255]
new_img[14, 58] = [0, 0, 255]
cv2.imwrite("./test.png", new_img)
