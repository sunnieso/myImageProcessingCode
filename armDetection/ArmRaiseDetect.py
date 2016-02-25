#!/usr/bin/env python

'''
Usage:
    rightArmDetect.py 
    OR,
    rightArmDetect.py {video filename}

take in a video of (a) human and detect the human's right arm movement.
Press any key to continue, ESC to stop.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
import cv2
import time

# define color
BLUE = (255,0,0)
RED = (0,0,255)
GREEN = (0,255,0)
ORANGE = (0,137,255)
NEON_BLUE = (255,100,180)
PINK = (222,0,255)

# define object
BODY = 1
HAND = 2

# declare video size
SCREEN_WIDTH = -1
SCREEN_HEIGHT = -1
SCREEN_SIZE = -1
# return true if rectangle r is inside rectangle q      
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
    white = (area / 4)/20
    # expand target width to make sure that it includes arms
    
    print("enter detectArm function")
    print("adjustedTarget = {}".format(adjustedTarget))
    white_locations = cv2.findNonZero(adjustedTarget)
    if((white_locations == None) or( len(white_locations) >= white)):
        print("detected right arm movement!!!")
        return True
    print("no arm movement")
    return False

def analyze(img,mask, deq, color1 = GREEN, color2 = RED, color3 = BLUE, object = BODY):
    new, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
    drawing = np.zeros(img.shape,np.uint8)

    max_area=-1
    handDetected = False;

    if (len(contours) != 0):       
        handDetected = False;
       #  find the biggest contour
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            # x,y,w,h = cv2.boundingRect(cnt)
            # if(object == BODY and h>w):
                # continue
            if(area>max_area):
                max_area=area
                ci=i
        # cnt is the biggest contour
        cnt=contours[ci]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:

            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
                 
            # draw contours 
            centr=(cx,cy)       
            cv2.circle(img,centr,3,[0,0,255],2)       
            cv2.drawContours(drawing,[cnt],0,color1,2) 
            cv2.drawContours(drawing,[hull],0,color2,2) 
            
            cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            hull = cv2.convexHull(cnt,returnPoints = False)
            
            # draw defects
            if(1):
                defects = cv2.convexityDefects(cnt,hull)
                mind=0
                maxd=0
                if( defects != None):
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        dist = cv2.pointPolygonTest(cnt,centr,True)
                        cv2.line(img,start,end,color1,2)                
                        cv2.circle(img,far,5,color2,-1)
                        print(i)
                        i=0

            # draw rectangle bounding the contour 
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            rec_area = h*w
            print("x,y = {},{}".format(x,y))
            print("area = {}".format(rec_area))
            cv2.rectangle(img, (x,y),(x+w,y+h) , color3)
            if(rec_area > 1000 and rec_area < 4000):
                # get the ratio of height vs width. a hand shouldn't be too flat or long
                if (h >= w):
                    ratio = h/(w+0.0)
                else:   
                    ratio = w/(h+0.0)
                print("ratio = {}".format(ratio))
                # 3 is the threshold I get after testing 
                if(ratio < 3):
                    handDetected = True;
                    cv2.putText(img, "hand", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color3, 2)
                    deq.append(centr)

            elif(h>w and rec_area > 6000 and rec_area < 15000):
                cv2.putText(img, "body", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color3, 2)
                  # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                        # cv2.putText(img,"Arm Raised", (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,2,(0,50,100))
                # cv2.circle() 
    if(handDetected):
        k = 0
        handRaised = 5
        for pt in deq:
            # print ("pt = {}".format(pt))
            if(k != 0 and deq[k][1] < deq[k-1][1]):
                handRaised -= 1
            k+=1
            cv2.circle(img,pt,k,RED,-1)
        if(handRaised <= 0):
            print("arm raised detected!!")
            cv2.putText(img, "Arm raised!", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color3, 3)
                
    return img


if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it
    from collections import deque 
    print(__doc__)

    if(len(sys.argv) != 1):
        try:
            cap = cv2.VideoCapture(sys.argv[1])
        except:
            # fail to read from input, read from webcam then.
            cap = cv2.VideoCapture(0)
    else:               
        cap = cv2.VideoCapture(0)
    # cv2.namedWindow('vid')
    cv2.namedWindow("contour")
    cv2.namedWindow('BackgroundSubtractor')
    cv2.namedWindow('threshold')
    cv2.moveWindow('BackgroundSubtractor',400,0)
    cv2.moveWindow('contour',0,0)
    cv2.moveWindow('threshold',0,400)
    # background subtractor. collect 500 frames.
    #                                           history,varTHreshold, bShadowDetection 
    fgbg = cv2.createBackgroundSubtractorMOG2( 300, 16, False)
    ret,img = cap.read()
    SCREEN_HEIGHT, SCREEN_WIDTH, im_channels = img.shape
    SCREEN_SIZE = SCREEN_WIDTH * SCREEN_HEIGHT
    print("w,h = {},{}".format(SCREEN_WIDTH,SCREEN_HEIGHT))
    # time.sleep(5)
    deq = deque([],10)
    while( cap.isOpened() ) :
        ret,img = cap.read()
        if(img == None):
            print("EOF")
            break
        img = cv2.resize(img, (320,240))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        # cv2.erode(blur,None)
        # cv2.dilate(blur,None)
        # blur = cv2.blur(gray,(5,5))
        fgmask = fgbg.apply(blur)
        # print ("len(fgmask) ={}".format(type(fgmask)))
        ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV)
        # print ("len(thresh1) ={}".format(type(thresh1)))
        mask2 = fgmask.copy()
        # contour_thres = analyze(img,thresh1,deq)
        contour_backg = analyze(img,mask2,deq, ORANGE,NEON_BLUE, PINK, object = HAND)
        
        cv2.imshow("contour", contour_backg)
        cv2.imshow("BackgroundSubtractor",fgmask)
        cv2.imshow("threshold",thresh1)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
