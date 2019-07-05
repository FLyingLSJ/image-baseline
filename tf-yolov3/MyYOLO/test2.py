import cv2
import numpy as np
import tensorflow as tf

a = np.random.randint(-10,10,(1,13,13,5))
wh = tf.constant(a,dtype=tf.float32)

sess = tf.Session()
print(sess.run(wh>0.0))
print(sess.run(tf.sqrt(wh)))
print(sess.run(tf.sqrt(tf.keras.backend.switch(wh>0.0,wh,tf.zeros_like(wh)))))
# image = cv2.imread("./girl.jpg")
# # cv2.imshow("img",image)
# # cv2.waitKey(0)
#
# print(image.shape)
# draw_1 = cv2.rectangle(image, (80, 30), (120, 80), (255, 0, 0), 2)
# print(draw_1.shape)
# cv2.imshow("img",draw_1)
# cv2.waitKey(0)
# # cv2.imwrite("vertical_flip.jpg", draw_1)