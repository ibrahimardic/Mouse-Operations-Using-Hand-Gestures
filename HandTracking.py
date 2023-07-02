import cv2
import mediapipe as mp
import math
from tkinter import *
import numpy


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): #Confidencelar %50 nin altına düserse tekrar ediyor. Böylelikle sistemin optimizasyonu hızlanıyor.
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        #Obje yaratacağımız için bu parametreleri belirliyoruz.

        self.mpHands = mp.solutions.hands #bu modeli kullanmadan önce yeni bir mpHands objesi belirliyoruz

        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img: object, draw: object = True) -> object:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv kutuphanesi RGB kullandığı için renkleri RGB'ye çeviriyoruz.
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks) #Eli algılayıp algılamamasını kontrol ettik

        if self.results.multi_hand_landmarks: #Eğer eli algılıyorsa
            for hands in self.results.multi_hand_landmarks:
                if draw: #Elin belirli yerlerine küçük kırmızı noktaları koyuyoruz, (Mediapipe'dan alıyor verileri)
                    self.mpDraw.draw_landmarks(img, hands,
                                               self.mpHands.HAND_CONNECTIONS)#Bu noktalar arasındaki edge'leri-bağlantıları çiziyor. (Yeşil noktalar).

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark): #Elimizdeki her noktayı numaralandırıyoruz.
                #print(id, lm)
                height, width, c = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)  #Elin koordinatlarını float değerden, int değere çeviriyoruz.
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy]) #Aldığımız bütün değerleri Landmark list'inin içine ekliyoruz ki sonradan bir parmağın verisini
                                                #buradan kolaylıkla çekebilelim.
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED) # Parmağımızdaki noktalara şekil verdik.

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
                #bbox'ı çizzdiğimiz yer, en küçük x ve y değerinin soluna ve aşağısına, en büyük x ve y değerinin sağına ve üstüne şeklinde çizdik




        return self.lmList, bbox



    def fingersUp(self):
        fingers = []
        # Thumb
        try:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        except:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            try:
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            except:
                fingers.append(0)

        #totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


    def findDistance2(self, p3, p4, img, draw=True, r=15, t=3):
        x0, y0 = self.lmList[p3][1:]
        x1, y1 = self.lmList[p4][1:]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        if draw:
            cv2.line(img, (x0, y0), (x1, y1), (255, 0, 255), t)
            cv2.circle(img, (x0, y0), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x1 - x0, y1 - y0)

        return length, img, [x0, y0, x1, y1, cx, cy]


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()     #kamera açmak için
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img) #Eldeki bütün noktaların koordinatlarını alıyoruz.
        if len(lmList) != 0: #Eğer lmList Array'inin içi boş değilse baş parmağımızın ucundaki noktanın koordinatını yazdırıyoruz.
            print(lmList[4])

        cv2.imshow("Hand Tracking", img)   #standart kamera açma işlemleri
        cv2.waitKey(1)



if __name__ == "__main__":
    main()