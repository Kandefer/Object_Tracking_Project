import numpy as np 
import cv2

import time
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

def klt(x, y, w, h, I, VOT_target):
    n_frame = 200
    frames = np.empty((n_frame,),dtype=np.ndarray)
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    # x = mouseX - kernel_size//2
    # y = mouseY - kernel_size//2
    # w = h = kernel_size
    # (x, y, w, h) = 160.0,52.0,53.0,177.0
    # (x, y, w, h) = cv2.selectROI("Select Object",I)
    # cv2.destroyWindow("Select Object")
    bboxs[0] = np.empty((1,4,2), dtype=float)
    bboxs[0][0,:,:] = np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]]).astype(float)
    print(np.shape(bboxs[0])[0])
    # Start from the first frame, do optical flow for every two consecutive frames.
    startXs,startYs = getFeatures(cv2.cvtColor(I,cv2.COLOR_RGB2GRAY),bboxs[0],use_shi=False)
    VOT = []
    for i in range(102,201):
        # nazwa_ze_sciezka = 'track_seq/track00' + str(i) + '.png'
        nazwa_ze_sciezka = '0_helikopter/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '1_dancer/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '2_lamb/track00' + str(i) + '.jpg'
        # nazwa_ze_sciezka = '3_diver/track00' + str(i) + '.jpg'
        print('Processing Frame',i)
        I_next = cv2.imread(nazwa_ze_sciezka)
        newXs, newYs = estimateAllTranslation(startXs, startYs, I, I_next)
        i = i-101
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])
        
        # update coordinates
        startXs = Xs
        startYs = Ys

        # update feature points as required
        n_features_left = np.sum(Xs!=-1)
        # print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(cv2.cvtColor(I_next,cv2.COLOR_RGB2GRAY),bboxs[i])

        # draw bounding box and visualize feature point for each object
        frames_draw[i] = I_next.copy()
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][0,:,:].astype(int))
        frames_draw[i] = cv2.rectangle(frames_draw[i], (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
        for k in range(startXs.shape[0]):
            frames_draw[i] = cv2.circle(frames_draw[i], (int(startXs[k,0]),int(startYs[k,0])),3,(0,0,255),thickness=2)
        x_vot, y_vot, w_vot, h_vot = VOT_target[i]
        if xmin+boxw<x_vot or ymin+boxh<y_vot or xmin>x_vot+w_vot or ymin>y_vot+h_vot:
            VOT.append(0)
        else:
            VOT.append(1)
        print(VOT)
        print(sum(VOT)/len(VOT))
        # imshow if to play the result in real time
        # cv2.imwrite('track_seq/resultKLT/result00' + str(i) + '.jpg', frames_draw[i])
        cv2.imwrite('0_helikopter/resultKLT/result00' + str(i) + '.jpg', frames_draw[i])
        # cv2.imwrite('1_dancer/resultKLT/result00' + str(i) + '.jpg', frames_draw[i])
        # cv2.imwrite('2_lamb/resultKLT/result00' + str(i) + '.jpg', frames_draw[i])
        # cv2.imwrite('3_diver/resultKLT/result00' + str(i) + '.jpg', frames_draw[i])
        cv2.imshow("win",frames_draw[i])
        # cv2.waitKey(10)
        k = cv2.waitKey(30)
        if k == 27:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(sum(VOT)/len(VOT))
