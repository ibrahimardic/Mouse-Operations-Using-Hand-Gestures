import cv2
import numpy
import HandTracking as ht
import time
import autopy
#from autopy.mouse import RIGHT
from tkinter import *
import pyautogui



wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7



pTime = 0
plocX, plocY = 0, 0 #Previous Locations
clocX, clocY = 0, 0 #Current Locations


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = ht.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)


while True:
    # 1. Ele çizdiğimiz işaretleri buluyoruz.
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. İşaret ve baş parmakların en üst noktalarını alıyoruz. (8,12)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

    # 3. Hangi parmakların yukarıda olduğuna bakıyoruz.
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)
    # 4. Eğer sadece işaret parmağı 1 değerindeyse,ve baş parmak ile orta parmak inikse mouse hareket modundadır.
    if (fingers[1] == 1 and fingers[2] == 0) and (fingers[1] == 1 and fingers[0] == 0) :
        # 5. Kordinasyonları dönüştürüyoruz.
        x3 = numpy.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = numpy.interp(y1, (frameR, hCam - frameR), (0, hScr))

        # Parmak hareketleri çok titrediği için değerleri yumuşatıyoruz.
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 6. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)

    #Eğer değeri Ekran boyutundan silmezsek
                                        #parmağımızı sola hareket ettirdiğimizde sağa hareket ediyordu.
                                        #Bu yüzden değerin tam tersini alarak bunu değiştirdik.

        cv2.circle(img, (x1, y1), 15, (0, 0, 0), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # 7. Eğer işaret ve baş parmak yukarıdaysa (1 değerindeyseler), tıklama moduna gir.
    if fingers[1] == 1 and fingers[2] == 1:
        # 8. İki parmağın uç değerleri arasındaki mesafeyi bul
        length, img, lineInfo = detector.findDistance(8, 12, img)
        #print(length)
        # 9. Eğer bu mesafe 30'dan düşükse tıkla.
        if length < 30:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                       15, (0, 255, 0), cv2.FILLED)
            coords=pyautogui.position()
            pyautogui.leftClick(coords)

    if fingers[0] == 1 and fingers[1] == 1 :

        # 11. Bu parmaklar arasındaki mesafeyi bul.
        length, img, lineInfo = detector.findDistance(4, 8, img)
        #print(length)
        # 10. Eğer mesafe kısaysa sağ click yap.

        if length < 30:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                       15, (0, 255, 0), cv2.FILLED)

            coords=pyautogui.position()
            pyautogui.click(coords, button="right",_pause=True)

    if cv2.waitKey(1) == ord('q'):  # Quit
        break

    cv2.imshow("Hand Tracking",img)
    cv2.waitKey(1)
