# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 17:29
# @Author  : ljf

import xml.etree.ElementTree as ET
import cv2
import glob


def show_box(xml_path):
    tree = ET.parse(xml_path)
    x1, y1, x2, y2 = 0, 0, 0, 0
    for elem in tree.iter():
        if elem.tag == "filename":
            img_path = "../data/aopti-8/images/{}".format(elem.text)
            print("图片路径：{}".format(img_path))
        elif elem.tag == "name":
            label = elem.text
            print("图片的label：{}".format(elem.text))
        elif elem.tag == "xmin":
            x1 = int(elem.text)
            print("box的左上角x坐标：{}".format(elem.text))
        elif elem.tag == "ymin":
            print("box的左上角y坐标：{}".format(elem.text))
            y1 = int(elem.text)
        elif elem.tag == "xmax":
            print("box的右下角x坐标：{}".format(elem.text))
            x2 = int(elem.text)
        elif elem.tag == "ymax":
            print("box的右下角y坐标：{}".format(elem.text))
            y2 = int(elem.text)
    image = cv2.imread("../data/aopti-8/images/11.tif")
    # print(image)
    # print(type(x1))
    cv2.putText(image, label, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1., (0, 0, 255), 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("box-image", image)
    cv2.waitKey(0)
if __name__ == "__main__":
    