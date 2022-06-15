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

cv2.imwrite("./base_map.png", new_img)


obstacles = np.empty((0, 2), dtype=int)
img = cv2.imread('base_map.png', cv2.IMREAD_COLOR)
for i in range(100):
    for j in range(100):
        if sum(img[i, j]) < 384:
            if i == 0:
                if j == 0:
                    if sum(img[i+1, j]) > 500 or sum(img[i, j+1]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)
                elif j == 99:
                    if sum(img[i+1, j]) > 500 or sum(img[i, j-1]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)
                else:
                    if sum(img[i+1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i, j-1]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)
            elif i == 99:
                if j == 0:
                    if sum(img[i-1, j]) > 500 or sum(img[i, j+1]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)
                elif j == 99:
                    if sum(img[i-1, j]) > 500 or sum(img[i, j-1]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)
                else:
                    if sum(img[i-1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i, j-1]) > 500:
                        obstacles = np.append(obstacles, [[i, j]], axis=0)
            elif j == 0:
                if sum(img[i+1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i-1, j]) > 500:
                    obstacles = np.append(obstacles, [[i, j]], axis=0)
            elif j == 99:
                if sum(img[i+1, j]) > 500 or sum(img[i, j-1]) > 500 or sum(img[i-1, j]) > 500:
                    obstacles = np.append(obstacles, [[i, j]], axis=0)

            elif sum(img[i+1, j]) > 500 or sum(img[i-1, j]) > 500 or sum(img[i, j+1]) > 500 or sum(img[i, j-1]) > 500:
                obstacles = np.append(obstacles, [[i, j]], axis=0)


new_img = np.zeros((100, 100, 3))
for i in range(100):
    for j in range(100):
        new_img[i,j]=[255,255,255]
for o in obstacles:
    new_img[o[0], o[1]] = [0, 0, 0]
cv2.imwrite("testest.png", new_img)
