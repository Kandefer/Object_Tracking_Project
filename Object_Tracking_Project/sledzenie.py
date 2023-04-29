import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from scipy.ndimage.measurements import center_of_mass as center_of_mass
from funkcje import MeanShiftFunc, CamShiftFunc, kcf
from KLT import klt
from PF import PF

def track_init(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.rectangle(param, (x-kernel_size//2, y-kernel_size//2), (x + kernel_size//2, y + kernel_size//2), (0, 255, 0), 2)
        mouseX, mouseY = x, y

# Wczytanie pierwszego obrazka
# I = cv2.imread('track_seq/track00100.png')
# kernel_size = 80 # rozmiar rozkladu
# mouseX, mouseY = (820, 430) # przykladowe wspolrzedne
# N = 8

I = cv2.imread('0_helikopter/track00101.jpg')
# (x, y, w, h) = 341,220,229,154
# kernel_size = 100 # rozmiar rozkladu
kernel_size = 80 # rozmiar rozkladu
mouseX, mouseY = (440, 140) # przykladowe wspolrzedne
N = 6
VOT_name = '0_helikopter/groundtruth.txt'

# I = cv2.imread('1_dancer/track00101.jpg')
# # kernel_size = 40 # rozmiar rozkladu
# # mouseX, mouseY = (190, 150) # przykladowe wspolrzedne
# (x, y, w, h) = 160,52,53,177
# kernel_size = 53 # rozmiar rozkladu
# mouseX, mouseY = (x + w//2, y + h//2) # przykladowe wspolrzedne
# N = 4
# VOT_name = '1_dancer/groundtruth.txt'

# I = cv2.imread('2_lamb/track00101.jpg')
# kernel_size = 100 # rozmiar rozkladu
# mouseX, mouseY = (640, 432) # przykladowe wspolrzedne
# N = 6

# I = cv2.imread('3_diver/track00101.jpg')
# (x, y, w, h) = 528,294,224,406
# kernel_size = 100 # rozmiar rozkladu
# mouseX, mouseY = (x + w//2, y + h//2) # przykladowe wspolrzedne
# N = 6
# VOT_name = '3_diver/groundtruth.txt'

x = mouseX - kernel_size//2
y = mouseY - kernel_size//2
w = h = kernel_size

cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', track_init, param=I)


# Pobranie klawisza
while(1):
    cv2.imshow('Tracking', I)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:# ESC
        break
    
with open(VOT_name) as f:
    lines = f.readlines()
VOT_target = []
for i, l in enumerate(lines):
    VOT_target.append(l.replace('\n', '').split(","))
    for j, s in enumerate(VOT_target[i]):
        VOT_target[i][j] = int(float(s))


# MeanShiftFunc(kernel_size, mouseX, mouseY, I, N, VOT_target)
# CamShiftFunc(x, y, w, h, I, VOT_target)
klt(x, y, w, h, I, VOT_target)
# kcf(x, y, w, h, I, VOT_target)
# PF(x, y, w, h, I, VOT_target)
