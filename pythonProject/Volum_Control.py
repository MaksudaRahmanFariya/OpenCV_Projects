import cv2
import mediapipe as mp
import time
import numpy as np
import Hand_Tracking_Module as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
###########################
wCam, wHig = 640, 480

#########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, wHig)
pTime = 0
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)
detector = htm.handDetector(detectionCon=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minVol = volumeRange[0]
maxVol = volumeRange[1]

while True:
    success, img = cap.read()
    #FindHand

    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=True)
    if len(lmlist) != 0:
        #Filter based on size
        area = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])//100
        #print(area)
        if 250 < area < 1000:
            # Find distance between index and thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)


            #convert volume
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])
            #Reduce resolution to make it smoother
            smootheness = 10
            volPer = smootheness * round(volPer/smootheness)
            #Check which finger is up
            #If pinky is down set volume
            fingers = detector.fingersUp()
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255,0,0)


            #Drawings
            # Frame rates

            # hand range 50-300
            # volume range -65-0

            # Reduce Resolution to make it smoother
            # smoothness = 10
            # volPer = smoothness * round(volPer / smoothness)
            # fingers = detector.fingersUp()
            #volume.SetMasterVolumeLevel(vol, None)
            # print(vol)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
        cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, colorVol, 3)

        # Frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cv2.putText(img,f'FPS: {int(fps)}', (55,78), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
