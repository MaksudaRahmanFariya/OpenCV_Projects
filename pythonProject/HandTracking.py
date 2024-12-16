import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()                           #this object only use RGB image
mpDraws = mp.solutions.drawing_utils              # for connecting landmarks
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlmk in results.multi_hand_landmarks:
            for  id,lm in enumerate(handlmk.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                #if id == 0:
                cv2.circle(img,(cx,cy),15, (255,0,255),cv2.FILLED)


            mpDraws.draw_landmarks(img, handlmk, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

    import cv2
    import mediapipe as mp
    import time


    class handDetection():

        def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
            self.mode = mode
            self.maxHands = maxHands
            self.detectionCon = detectionCon
            self.trackCon = trackCon
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon,
                                            self.trackCon)  # this object only use RGB image
            self.mpDraws = mp.solutions.drawing_utils  # for connecting landmarks

        def findhand(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            # print(results.multi_hand_landmarks)
            if self.results.multi_hand_landmarks:
                for handlmk in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraws.draw_landmarks(img, handlmk, self.mpHands.HAND_CONNECTIONS)
            return img  # print(id,lm)

        def findposition(self, img, handNo=0, draw=True):
            lmklist = []
            if self.results.multi_hand_landmarks:
                handlmk = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(handlmk.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmklist.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return lmklist

            # mpDraws.draw_landmarks(img, handlmk, mpHands.HAND_CONNECTIONS)

        # print(results.multi_hand_landmarks)


    def main():
        pTime = 0
        cTime = 0
        cap = cv2.VideoCapture(0)
        detector = handDetection()
        while True:
            success, img = cap.read()
            img = detector.findhand(img)
            lmklist = detector.findposition(img)
            if len(lmklist) != 0:
                print(lmklist[4])
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)


    if __name__ == "__main__":
        main()