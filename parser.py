import tkinter as tk
from tkinter import filedialog
import mapper
import os
import cv2
import numpy as np
import pytesseract

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    root = tk.Tk()
    root.withdraw()
    readImg = filedialog.askopenfile(initialdir=os.path.expanduser(r'~\Documents\receipt'))
    readImg.close()

    orig = cv2.imread(readImg.name)
    # orig = cv2.resize(orig, (475, 700))
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    value = 50
    mask = (255 - gray) < value
    gray_new = np.where((255 - gray) < value, 255, gray + value)
    thresh, bw = cv2.threshold(gray_new, 177, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    ret, thresh = cv2.threshold(bw, 177, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # retrieve the contours as a list, with simple apprximation model
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # the loop extracts the boundary contours of the page
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break
    approx = mapper.mapp(target)  # find endpoints of the sheet
    w = 1500
    h = 3000
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # map to 800*800 target window

    op = cv2.getPerspectiveTransform(approx, pts)  # get the top or bird eye view effect
    dst = cv2.warpPerspective(bw, op, (w, h))

    resized = cv2.resize(dst, (200, 550))
    cv2.imshow("Scanned", resized)
    print(pytesseract.image_to_string(dst))

    cv2.imwrite('scanned.png', dst)
    # cv2.imshow('blur', orig)
    cv2.waitKey(0)

    # s = input("Press enter to exit")
    quit()
