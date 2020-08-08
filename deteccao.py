import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class deteccaoMoedas:
    @staticmethod
    def detect(img):
        print("[INFO] detectando moedas na imagem...")
        img_output = img.copy()
        img_masks = img.copy()
        moedas=[]

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        gaussian = cv.GaussianBlur(img_gray, (5, 5), 0) #5 5

        canny = cv.Canny(gaussian, 40, 100) #100 200

        rows = canny.shape[0]
        circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=20, minRadius=20, maxRadius=300)
     
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv.circle(img_output, center, 1, (0, 100, 100), 3)
                radius = i[2]
                cv.circle(img_output, center, radius, (255, 0, 255), 3)

                crop = img_masks[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
                crop = cv.resize(crop,(96,96))
                moedas.append(crop)
        print("[INFO] quantidade de moedas encontradas: {}.".format(len(moedas)))
        return moedas