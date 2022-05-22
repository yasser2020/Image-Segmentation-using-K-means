# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:14:32 2022

@author: Yasser Ezzat
"""

import numpy as np
import cv2
img=cv2.imread('data/BSE_Image.jpg')
img2=img.reshape((-1,3))
img2=np.float32(img2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#Clusters
k=4
attempts=10
ret,label,center=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center=np.uint8(center)

res=center[label.flatten()]
res2=res.reshape((img.shape))
cv2.imwrite('data/segmented.jpg', res2)
 