import numpy as np
import cv2 as cv

images = []
with open("Images.txt", 'w') as f:
    for i in range(10):
        img = cv.imread('images/'+str(i)+'.png')
        for row in img:
            for col in row:
                f.write(str(col[0]))
                f.write(',')
        f.write('\n')

