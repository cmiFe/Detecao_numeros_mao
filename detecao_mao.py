import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)

#def nothing(x):
 #   pass
#cv2.namedWindow("Trackbars")
#cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
#cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
#cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
#cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
#cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
#cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)



while True:
        
    try:
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        roi=frame[60:330, 60:330]
        
        
        cv2.rectangle(frame,(60,60),(330,330),(0,255,0),0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ## CASO QUEIRA RELUGAR O HSV BASTA RETIRAR DE COMENTARIO
        # ESSAS LINHAS E OS OBJETOS CRIADOS POR ELA NA PARTE DE CIMA DO CODIGO
        #l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        #l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        #l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        #u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        #u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        #u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        lower_blue = np.array([0, 23, 30])
        upper_blue = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        mask = cv2.medianBlur(mask,21) 
        _,contours,hierarchy= cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        epsilon = 0.0001*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
        hull = cv2.convexHull(cnt)
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
        arearatio=((areahull-areacnt)/areacnt)*100
        hull = cv2.convexHull(approx, returnPoints=False)
        defeitos = cv2.convexityDefects(approx, hull)
        l=0
        for i in range(defeitos.shape[0]):
            s,e,f,d = defeitos[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            d=(2*ar)/a
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            cv2.line(roi,start, end, [255,255,0], 2)
        l+=1
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Area Vazia',(0,50), fonte, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<12:
                    cv2.putText(frame,'0',(0,50), fonte, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame,'1',(0,50), fonte, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==2:
            cv2.putText(frame,'2',(0,50), fonte, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
            
            cv2.putText(frame,'3',(0,50), fonte, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            cv2.putText(frame,'4',(0,50), fonte, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            cv2.putText(frame,'5',(0,50), fonte, 2, (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow('hsv',hsv)
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)

    except:
        pass   

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
