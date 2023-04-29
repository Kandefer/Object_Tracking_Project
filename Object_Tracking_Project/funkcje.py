import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from scipy.ndimage.measurements import center_of_mass as center_of_mass

def gen_gauss(kernel_size, mouseX, mouseY, I, N):
    # Generowanie Gaussa
    sigma = kernel_size/N  # odchylenie std
    x = np.arange(0, kernel_size, 1, float)  # wektor poziomy
    y = x[:, np.newaxis]  # wektor pionowy
    x0 = y0 = kernel_size // 2  # wsp. srodka
    G = 1 / (2 * math.pi * sigma ** 2) * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    xS = mouseX - kernel_size // 2
    yS = mouseY - kernel_size // 2

    I_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    I_H = I_HSV[:, :, 0]
    hist_q = np.zeros((256, 1), float)
    for jj in range(0, kernel_size):
        for ii in range(0, kernel_size):
            pixel_H = I_H[yS + jj, xS + ii]
            hist_q[pixel_H] += G[jj, ii]

    hist_q = hist_q / np.amax(hist_q)
    I_H_part = I_H[yS : yS + kernel_size, xS : xS + kernel_size]
    return hist_q, I_H_part

def compute_moments(img, i, j):
    moment = 0
    for x, row in enumerate(img):
        for y, pixel in enumerate(row):
            moment = moment + ( x**i*y**j * pixel )
    return moment

def MeanShiftFunc(kernel_size, mouseX, mouseY, I, N, VOT_target):
    hist_q_1, I_H_1 = gen_gauss(kernel_size=kernel_size, mouseX=mouseX, mouseY=mouseY, I=I, N=N)
    yS = mouseY
    xS = mouseX
    
    VOT = []
    for i in range(101, 201):
        # nazwa_ze_sciezka = 'track_seq/track00' + str(i) + '.png'
        # nazwa_ze_sciezka = '0_helikopter/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '1_dancer/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '2_lamb/track00' + str(i) + '.jpg'
        nazwa_ze_sciezka = '3_diver/track00' + str(i) + '.jpg'
        
        I_next = cv2.imread(nazwa_ze_sciezka)
        cv2.rectangle(I_next, (xS - kernel_size // 2, yS - kernel_size // 2), (xS + kernel_size // 2, yS + kernel_size // 2),
                      (0, 255, 0), 2)
    
        hist_q_2, I_H_2 = gen_gauss(kernel_size=kernel_size, mouseX=xS, mouseY=yS, I=I_next, N=N)
    
        x_na_potem = xS - kernel_size // 2
        y_na_potem = yS - kernel_size // 2
    
        bhatta = np.sqrt(hist_q_2 * hist_q_1)
        # plt.figure()
        # plt.imshow(I_H_1)
        # plt.figure()
        # plt.imshow(I_H_2)
        # plt.show()
        
        I = I_next
        # hist_q_1 = hist_q_2
    
        Wynik = np.zeros((kernel_size, kernel_size), dtype=float)
        for jj in range(0, kernel_size):
            for ii in range(0, kernel_size):
                pixel_H = I_H_2[jj, ii]
                Wynik[jj, ii] = bhatta[pixel_H] * hist_q_2[pixel_H]
                # print(Wynik[jj, ii])
        #srodek ciezkości to suma po x i po y i dzielimy po wartosci calego wycinka
        xS, yS = center_of_mass(Wynik)
        xS = int(xS + x_na_potem)
        yS = int(yS + y_na_potem)
    
        # print(xS, yS)
    
        # plt.figure()
        # plt.imshow(I_next)
        # plt.plot(xS, yS, '*m')
        # plt.show()
        # cv2.imwrite('track_seq/result/result00' + str(i) + '.jpg', I_next)
        # cv2.imwrite('0_helikopter/result/result00' + str(i) + '.jpg', I_next)
        # cv2.imwrite('1_dancer/result/result00' + str(i) + '.jpg', I_next)
        # cv2.imwrite('2_lamb/result/result00' + str(i) + '.jpg', I_next)
        cv2.imwrite('3_diver/result/result00' + str(i) + '.jpg', I_next)
        cv2.imshow('Tracking', I_next)
        x, y, w, h = xS - kernel_size // 2, yS - kernel_size // 2, kernel_size, kernel_size
        x_vot, y_vot, w_vot, h_vot = VOT_target[i-100]
        if x+w<x_vot or y+h<y_vot or x>x_vot+w_vot or y>y_vot+h_vot:
            VOT.append(0)
        else:
            VOT.append(1)
        print(VOT)
        print(sum(VOT)/len(VOT))
        
        k = cv2.waitKey(30)
        if k == 27:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CamShiftFunc(x, y, w, h, I, VOT_target):
    # x = mouseX - kernel_size//2
    # y = mouseY - kernel_size//2
    # w = h = kernel_size
    # (x, y, w, h) = 160,52,53,177
    frame = I
    track_window = (x, y, w, h//2)
    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    VOT = []
    for i in range(101, 201):
        nazwa_ze_sciezka = 'track_seq/track00' + str(i) + '.png'
        # nazwa_ze_sciezka = '0_helikopter/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '1_dancer/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '2_lamb/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '3_diver/track00' + str(i) + '.jpg'
        frame = cv2.imread(nazwa_ze_sciezka)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imwrite('track_seq/resultCam/result00' + str(i) + '.jpg', img2)
        # cv2.imwrite('0_helikopter/resultCam/result00' + str(i) + '.jpg', img2)
        # cv2.imwrite('1_dancer/resultCam/result00' + str(i) + '.jpg', img2)
        # cv2.imwrite('2_lamb/resultCam/result00' + str(i) + '.jpg', img2)
        # cv2.imwrite('3_diver/resultCam/result00' + str(i) + '.jpg', img2)
        cv2.imshow('img2',img2)
        x, y, w, h = track_window
        x_vot, y_vot, w_vot, h_vot = VOT_target[i-100]
        if x+w<x_vot or y+h<y_vot or x>x_vot+w_vot or y>y_vot+h_vot:
            VOT.append(0)
        else:
            VOT.append(1)
        print(VOT)
        print(sum(VOT)/len(VOT))
        k = cv2.waitKey(30)
        if k == 27:
            break
        # if cv.waitKey(1) == ord('q'):
        #     break
        # cv.waitKey(0)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
from kcf import Tracker

def kcf(x, y, w, h, I, VOT_target):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("video", help="video you want to track", type=str)
    # args = parser.parse_args()
    # print(args)
    # cap = cv2.VideoCapture(args.video)
    tracker = Tracker()
    frame = I
    # roi = cv2.selectROI("tracking", frame, False, False)
    # x = mouseX - kernel_size//2
    # y = mouseY - kernel_size//2
    # w = h = kernel_size
    # (x, y, w, h) = 160,52,53,177
    roi = (x, y, w, h)
    tracker.init(frame, roi)
    VOT = []
    for i in range(101,201):
        # nazwa_ze_sciezka = 'track_seq/track00' + str(i) + '.png'
        nazwa_ze_sciezka = '0_helikopter/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '1_dancer/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '2_lamb/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '3_diver/track00' + str(i) + '.jpg'
        frame = cv2.imread(nazwa_ze_sciezka)
        x, y, w, h = tracker.update(frame)
        # x, y, w, h = VOT_target[i-100]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imwrite('track_seq/resultKFC/result00' + str(i) + '.jpg', frame)
        cv2.imwrite('0_helikopter/resultKFC/result00' + str(i) + '.jpg', frame)
        # cv2.imwrite('1_dancer/resultKFC/result00' + str(i) + '.jpg', frame)
        # cv2.imwrite('2_lamb/resultKFC/result00' + str(i) + '.jpg', frame)
        # cv2.imwrite('3_diver/resultKFC/result00' + str(i) + '.jpg', frame)
        x_vot, y_vot, w_vot, h_vot = VOT_target[i-100]
        if x+w<x_vot or y+h<y_vot or x>x_vot+w_vot or y>y_vot+h_vot:
            VOT.append(0)
        else:
            VOT.append(1)
        print(VOT)
        print(sum(VOT)/len(VOT))
        # print(x, y, w, h)
        cv2.imshow('tracking', frame)
        k = cv2.waitKey(30)
        if k == 27:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()