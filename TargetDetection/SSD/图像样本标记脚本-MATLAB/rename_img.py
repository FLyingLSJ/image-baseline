#encoding=utf-8
import os
import numpy as np
import re
import pandas as pd
import cv2
dir_img=os.getcwd()
pos = 6
out_img=dir_img+os.sep+"renamed_img"+os.sep
dir_img=dir_img+os.sep+"source_img"+os.sep
for i,img in enumerate(os.listdir(dir_img)):
    
        img=cv2.imread(dir_img+img,cv2.IMREAD_COLOR)
        #height, width = img.shape[:2]
        size = (500, 375)
        new_img=cv2.resize(img,size,interpolation=cv2.INTER_AREA)
        #cv2.namedWindow('leqna',cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('lena',new_img)
        #cv2.waitKey()
        c = str(i)
        ze = pos-len(c)
        name = '0'*ze+str(i)    
        name=name+".jpg"
        cv2.imwrite(out_img+name, new_img)
print("processed all images")