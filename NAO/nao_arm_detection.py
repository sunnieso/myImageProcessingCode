''' this is used in Choreographe python script box
'''
import numpy as np
import cv2
import time
import sys
from glob import glob
import itertools as it
from collections import deque

class MyClass(GeneratedClass):
    # define drawing colors
    BLUE = (255,0,0)
    RED = (0,0,255)
    GREEN = (0,255,0)
    ORANGE = (0,137,255)
    NEON_BLUE = (255,100,180)
    PINK = (222,0,255)
    # define object
    BODY = 1
    HAND = 2

    def __init__(self):
        GeneratedClass.__init__(self)

    def connect2Camera(self):
        try:
            self.cam = ALProxy("ALVideoDevice")
            clientName = self.getName()
            nCameraNum = 0
            nResolution = 1
            nColorspace = 0 #0 = luma , 13 = BGR # 11 = rgb
            nFps = 5
            self.client = self.cam.subscribeCamera(clientName,nCameraNum, nResolution, nColorspace, nFps)
        except BaseException, err:
            self.log("[ERR]Connect2Camera: %s" % err)

    def disconnectFromCamera(self):
        try:
            self.cam.unsubscribe(self.client)
        except BaseException, err:
            self.log("[ERR]disconnectFromCamera: %s" % err)


    def getImageFromCamera(self):
        try:
            self.dataImg = self.cam.getImageRemote(self.client)
            if(self.dataImg != None):
                self.img = np.reshape(np.frombuffer(self.dataImg[6], dtype='%iuint8' % self.dataImg[2]), (self.dataImg[1],self.dataImg[0],self.dataImg[2]))
                return self.img
        except BaseException,err:
            self.log("[ERR]getImageFromCamera: %s" % err)
        return None


    # detectArm detect right arm only
    # fgmask = the image after apply background subtractor
    # target = the vector that define the area of the "people body" [[x,y,w,h]]
    # area = the size of target area (int)
    def analyze(self, img, mask, deq, color1 = GREEN, color2 = RED, color3 = BLUE, imgc = 0, object = BODY):
        self.contours, self.hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        self.drawing = np.zeros(img.shape,np.uint8)

        self.max_area=-1
        self.handDetected = False;

        if (len(self.contours) != 0):
            self.handDetected = False;
           #  find the biggest contour
            for self.i in range(len(self.contours)):
                self.cnt=self.contours[self.i]
                self.area = cv2.contourArea(self.cnt)
                if(self.area>self.max_area):
                    self.max_area=self.area
                    self.ci=self.i
            # cnt is the biggest contour
            self.cnt=self.contours[self.ci]
            self.hull = cv2.convexHull(self.cnt)
            self.moments = cv2.moments(self.cnt)
            if self.moments['m00'] != 0:

                self.cx = int(self.moments['m10']/self.moments['m00']) # cx = M10/M00
                self.cy = int(self.moments['m01']/self.moments['m00']) # cy = M01/M00

                # draw contours
                self.centr=(self.cx,self.cy)
                cv2.circle(img,self.centr,3,[0,0,255],2)
                cv2.drawContours(self.drawing,[self.cnt],0,color1,2)
                cv2.drawContours(self.drawing,[self.hull],0,color2,2)

                self.cnt = cv2.approxPolyDP(self.cnt,0.01*cv2.arcLength(self.cnt,True),True)
                self.hull = cv2.convexHull(self.cnt,returnPoints = False)

                # draw defects
                if(len(self.hull) >= 4):
                    self.defects = cv2.convexityDefects(self.cnt,self.hull)
                    self.mind=0
                    self.maxd=0
                    if( self.defects != None):
                        for self.i in range(self.defects.shape[0]):
                            self.s,self.e,self.f,self.d = self.defects[self.i,0]
                            self.start = tuple(self.cnt[self.s][0])
                            self.end = tuple(self.cnt[self.e][0])
                            self.far = tuple(self.cnt[self.f][0])
                            self.dist = cv2.pointPolygonTest(self.cnt,self.centr,True)
                            cv2.line(img,self.start,self.end,color1,2)
                            cv2.circle(img,self.far,5,color2,-1)
#                            self.log(self.i)
                            self.i=0

                # draw rectangle bounding the contour
                self.rect = cv2.boundingRect(self.cnt)
                self.x,self.y,self.w,self.h = self.rect
                self.rec_area = self.h*self.w
#                self.log("x,y = {},{}".format(self.x,self.y))
#                self.log("area = {}".format(self.rec_area))
                cv2.rectangle(img, (self.x,self.y),(self.x+self.w,self.y+self.h) , color3)

                # method 1
                # if contour area is within 1000~4000, then we define this area to be hand/arm
                # store this position to the arm queue, and check if the arm position is gradually moving up.
                if(self.rec_area > 1000 and self.rec_area < 4000):
                    # get the ratio of height vs width. a hand shouldn't be too flat or long
                    if (self.h >= self.w):
                        self.ratio = self.h/(self.w+0.0)
                    else:
                        self.ratio = self.w/(self.h+0.0)
#                    self.log("ratio = {}".format(self.ratio))
                    # 3 is the threshold I get after testing
                    if(self.ratio < 3):
                        self.handDetected = True;
                        cv2.putText(img, "hand", (self.x,self.y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color3, 2)
                        deq.append(self.centr)

                # method 2
                # if contour area falls into the defined range 6000~ 15000, then we say this is human body.
                # check if the upper right side of the body image has a lot of black pixels,
                # if there is enough black pixels, then we say that these black pixels are the spaces between
                # raising arm and body. Therefore, we asumme that the arm is raised.
                elif(self.h>self.w and self.rec_area > 6000 and self.rec_area < 15000):
                    cv2.putText(img, "body", (self.x,self.y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color3, 2)
#                    # divide body rec to four parts.(our interests is upper left)
#                    self.target = img[self.y:self.y+(self.h/2), self.x:self.x+(self.w/2)]
#                    cv2.rectangle(img, (self.x, self.y),( self.x + (self.w/2), self.y + (self.h/2)),self.RED,1)
#                    self.white = self.rec_area / 4
#                    self.white_locations = cv2.findNonZero(self.target)
#                    if((self.white_locations == None) or( len(self.white_locations) <= self.white)):
#                        cv2.putText(img, "Arm raised!", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color3, 3)



        if(self.handDetected):
            self.k = 0
            self.handRaised = 5
            for self.pt in self.deq:
                self.log ("pt = {}".format(self.pt))
                if(self.k != 0 and self.deq[self.k][1] < self.deq[self.k-1][1]):
                    self.handRaised -= 1
                self.k+=1
                cv2.circle(img,self.pt,self.k,self.RED,-1)
            if(self.handRaised <= 0):
                self.log("arm raised detected!!")
                cv2.putText(img, "Arm raised!", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color3, 3)
                cv2.imwrite('/home/nao/recordings/cameras/armRaised_'+str(imgc)+'.jpg', img)
                return True
        return False





    def Main(self):
        self.awareness.stopAwareness()
#        self.say.say('Hi, it is time to do some good workout. Can you raise your arm like this?')
        self.motion.setBreathEnabled('Body', False)
        self.motion.stiffnessInterpolation("Head", 1.0, 0.5)


        # We set the fraction of max speed
        self.pMaxSpeedFraction = 0.2
        #set head to look up
        self.motion.angleInterpolationWithSpeed("HeadPitch", -25.0*almath.TO_RAD, self.pMaxSpeedFraction)
        self.motion.angleInterpolationWithSpeed("HeadYaw", 0.0*almath.TO_RAD, self.pMaxSpeedFraction)

        self.connect2Camera()

        self.fgbg = cv2.BackgroundSubtractorMOG( 300, 16, False)

        self.deq = deque([],10)
        self.imgc = 0
        self.running = True
        self.ret = False
        while( self.running and not self.ret ) :
#            self.log("looping...")
            time.sleep(0.2)
            self.frame = self.getImageFromCamera()
            if(self.frame == None):
                self.log("EOF")
                break
            self.blur = cv2.GaussianBlur(self.frame,(5,5),0)
            self.fgmask = self.fgbg.apply(self.blur)
#            cv2.imwrite('/home/nao/recordings/cameras/bg_'+str(self.imgc)+'.jpg', self.fgmask)
            self.ret = self.analyze(self.frame,self.fgmask,self.deq, self.ORANGE, self.NEON_BLUE, self.PINK,imgc = self.imgc, object = self.HAND)
            self.imgc += 1


        self.awareness.startAwareness()
        self.motion.setBreathEnabled('Body', True)
        self.disconnectFromCamera()
        self.log('end of main')

    def onLoad(self):
        #put initialization code here
        self.leds = ALProxy("ALLeds")
        self.photoCaptureProxy = ALProxy("ALPhotoCapture")
        self.say = ALProxy("ALTextToSpeech")
        self.awareness = ALProxy("ALBasicAwareness")
        self.motion = ALProxy("ALMotion")

    def onUnload(self):
        #put clean-up code here
        self.running = False
        pass

    def onInput_onStart(self):
        self.Main()
        self.running = False
        self.onStopped()

    def onInput_onStop(self):
        self.log("onstop")
        self.onUnload() #it is recommended to reuse the clean-up as the box is stopped
        self.onStopped() #activate the output of the box