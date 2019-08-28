#encoding=utf-8

import sys
import os
import codecs
import cv2

root = 'D:\\code\\fasterrcnn\\xml'
fp = open('D:\\code\\fasterrcnn\\output_img_label\\pos.txt','r')
fp2 = open('train.txt', 'w')
uavinfo = fp.readlines()

for i in range(len(uavinfo)):
    line = uavinfo[i]
    line = line.strip().split('\t')   
    img = cv2.imread(line[0])
    sp = img.shape
    height = sp[0]
    width = sp[1]
    depth = sp[2]
    info1 = line[0].split('\\')[-1]
    info2 = info1.split('.')[0]

    l_pos1 = line[1]
    l_pos2 = line[2]
    r_pos1 = line[3]
    r_pos2 = line[4]
    if int(r_pos1)>int(height):
        r_pos1=str(int(height)-1)
    if int(r_pos2)>int(width):
        r_pos2=str(int(width)-1)
    fp2.writelines(info2 + '\n')
    with codecs.open(root +r'\\'+ info2 + '.xml', 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
        xml.write('\t<filename>' + info1 + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>my database</database>\n')
        xml.write('\t\t<annotation>voc2007</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>null</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>XuSenhai</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        xml.write('\t<object>\n')
        xml.write('\t\t<name>uav</name>\n')
        xml.write('\t\t<pose>Unspecified</pose>\n')
        xml.write('\t\t<truncated>0</truncated>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<bndbox>\n')
        xml.write('\t\t\t<xmin>' + l_pos1 + '</xmin>\n')
        xml.write('\t\t\t<ymin>' + l_pos2 + '</ymin>\n')
        xml.write('\t\t\t<xmax>' + r_pos1 + '</xmax>\n')
        xml.write('\t\t\t<ymax>' + r_pos2 + '</ymax>\n')
        xml.write('\t\t</bndbox>\n')
        xml.write('\t</object>\n')
        xml.write('</annotation>')
fp2.close()