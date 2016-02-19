#!/usr/bin/env python

'''
example to detect upright people in images using HOG features

Usage:
    peopledetect.py <image_names>

Press any key to continue, ESC to stop.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
import cv2
import time


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1, color = (0,255,0)):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), color, thickness)

# detectArm detect right arm only
# fgmask = the image after apply background subtractor 
# target = the vector that define the area of the "people body" [[x,y,w,h]]
# area = the size of target area (int)
def detectArm(adjustedTarget,target,area):
    # extract the right arm 
    # number of pixel that needs to be white:
    white = area / 4
    # expand target width to make sure that it includes arms
    
    print("enter detectArm function")
    print("adjustedTarget = {}".format(adjustedTarget))
    white_locations = cv2.findNonZero(adjustedTarget)
    if(len(white_locations) >= white):
        print("detected right arm movement!!!")
        return True
    print("no arm movement")
    return False


if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

    print(__doc__)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    cap = cv2.VideoCapture(0)
    # self.prev = self.getImageFromCamera()
    cv2.namedWindow('vid')
    cv2.namedWindow("After NMS")
    cv2.namedWindow('BackgroundSubtractor')
    # default = ['../data/basketball2.png '] if len(sys.argv[1:]) == 0 else []
    k=0
    # background subtractor. collect 500 frames.
    #                                           history,varTHreshold, bShadowDetection 
    fgbg = cv2.createBackgroundSubtractorMOG2( 300, 16, True)

    while(True):
        ret, img = cap.read()
        orig = img.copy()

        cv2.imshow('vid',img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # cv2.imwrite('./tmpIm/img_' + str(k) + '.jpg', img)
        # k += 1 
        # time.sleep(1)
        # if(k >= 10):
        #     break
        # continue


        found, w = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        # draw_detections(orig, found_filtered, 3)
        
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        # found_filtered = np.array([[x, y, x + w, y + h] for (x, y, w, h) in found_filtered])
        # pick = non_max_suppression(found_filtered, probs=None, overlapThresh=1)
        # get background subtractor at right half of the target
        fgmask = fgbg.apply(orig)

        if len(found_filtered):
            target = [max(found_filtered, key = lambda x: x[2] * x[3])]
            area = target[0][2] * target[0][3]
            print("area = {}".format(target[0][2] * target[0][3]))
            if area > 30000:
                draw_detections(img, target, 3, (0,0,255))
                # get the area for arm 
                # try:
                x1 = target[0][0] #- target[0][2]/2
                x2 = target[0][0] + target[0][2]/2
                y1 = target[0][1]
                y2 = target[0][1] + target[0][3]/2
                # adjustedTarget = fgmask[(target[0][0] - target[0][2]/2): target[0][1], target[0][2]/2, target[0][3]/2]
                adjustedTarget = fgmask[y1:y2, x1:x2]
                cv2.imwrite("~/tmpIm/adjustedTarget.jpg", adjustedTarget)
                draw_detections(img,[[x1,y1,x2-x1,y2-y1]], 1, (0,50,100))

                if(detectArm(adjustedTarget,target,area)):

                    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                    cv2.PutText(img,"Arm Raised", (adjustedTarget[0],adjustedTarget[1]),cv2.FONT_HERSHEY_SIMPLEX,2,(0,50,100))
                else: 
                    print("detectArm fails. invalid target located at the side")
                    

        cv2.imshow('BackgroundSubtractor',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        # show some information on the number of bounding boxes
        # print("[INFO] {} original boxes, {} after suppression. ".format(
            # len(found), len(found_filtered)))

        # show the output images
        cv2.imshow('vid', orig)
        cv2.imshow("After NMS", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
