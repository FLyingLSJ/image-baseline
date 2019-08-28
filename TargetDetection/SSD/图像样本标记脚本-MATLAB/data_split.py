#coding=utf-8
import cv2
import os
import random
#test=open('D:\\code\\SSD\\VOCtrainval_06-Nov-2007\\VOC2007\\ImageSets\\Main\\aeroplane_train.txt','r',encoding='utf8')
#for line in test:
    #print(line)
fp = open('D:\\code\\fasterrcnn\\output_img_label\\pos.txt','r')
root = 'D:\\code\\fasterrcnn\\VOC2007\\ImageSets\\Main'
#fp = open(root + '\\'+'name_list.txt')
fp_trainval = open(root + '\\'+'trainval.txt', 'w')
fp_test = open(root + '\\'+'test.txt', 'w')
fp_train = open(root + '\\'+'train.txt', 'w')
fp_val = open(root + '\\'+'val.txt', 'w')

filenames = fp.readlines()
for i in range(len(filenames)):
    pic_name = filenames[i]
    pic_name = pic_name.split('\t')[0]
    x = random.uniform(0, 1)
    pic_info = pic_name.split('\\')[-1].split('.')[0]

    if x >= 0.5:
        fp_trainval.writelines(pic_info +" "+str(1)+ '\n')

    else:
        fp_test.writelines(pic_info +" "+str(1)+ '\n')

fp_trainval.close()
fp_test.close()



fp = open(root + '\\' +'trainval.txt','r')
filenames = fp.readlines()
for i in range(len(filenames)):
    pic_name = filenames[i]
    pic_name = pic_name.strip()
    pic_info = pic_name.split('.')[0]
    x = random.uniform(0, 1)
    if x >= 0.5:
        fp_train.writelines(pic_info +" "+str(1)+ '\n')
    else:
        fp_val.writelines(pic_info +" "+str(1)+ '\n')
fp_train.close()
fp_val.close()