from data_collection.img_process import grab_screen
import cv2
import time
from data_collection.key_cap import key_check
import numpy as np


# num1 573:581, 683:688
# num2 573:581, 690:695
# num3 573:581, 697:702

def main():
    numbers = []
    record = False
    i = 0
    empty = 0

    while True:
        time.sleep(0.2)

        keys = key_check()
        if 'T' in keys:
            record = not record
        elif 'Z' in keys:
            np.save('digits.npy', numbers)

        image = grab_screen("Grand Theft Auto V")

        # speed
        vis = image[568:577, 680:699, :]

        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
        # ret, vis = cv2.threshold(vis, 140, 255, cv2.THRESH_BINARY_INV)
        # ret, vis = cv2.threshold(vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vis = cv2.adaptiveThreshold(vis, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -5)

        if record:
            i += 1
            print(i)
            numbers.append(vis[:, :5])      # num1
            numbers.append(vis[:, 7:12])    # num2
            if empty % 5 == 0:
                numbers.append(vis[:, -5:])     # num3
            empty += 1

        vis = cv2.resize(vis, None, fx=10, fy=10)
        cv2.imshow("Frame", vis)
        key = cv2.waitKey(1) & 0xFF
        # cv2.waitKey()

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


if __name__ == '__main__':
    main()
