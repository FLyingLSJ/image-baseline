# coding: utf-8
import os
import cv2
import numpy as np
import tensorflow as tf
from models.yolo.config import config as cfg
from models.yolo.net import darknet53

sess = tf.Session()


class YOLOV3(object):

    def __init__(self, net, labels, batch_size, is_training=True):
        self.num_class = len(cfg.CLASSES)
        self.batch_size = batch_size
        self.ignore_thresh = cfg.IGNORE_THRESH

        # net config
        self.image_size = net.image_size
        self.cell_size = net.cell_size  # 包含特征图大小的列表
        self.scale = self.image_size / self.cell_size  # 图像到特征图的压缩比
        self.num_anchors = net.num_anchors
        self.anchors = net.anchors
        self.anchor_mask = net.anchor_mask
        self.x_scale = net.x_scale
        self.y_scale = net.y_scale

        # loss config
        self.object_alpha = cfg.OBJECT_ALPHA
        self.no_object_alpha = cfg.NO_OBJECT_ALPHA
        self.class_alpha = cfg.CLASS_ALPHA
        self.coord_alpha = cfg.COORD_ALPHA

        # 全部的损失值
        self.loss = 0.0
        if is_training:
            self.scales = net.get_output()
            self.total_loss(labels)
        else:
            self.scales = net.get_output()

    def calculate_object_confidence_loss(self, object_confidence_hat, object_confidence, object_mask,
                                         best_confidence_mask, scope='object_confidence_loss'):
        with tf.name_scope(scope):
            object_loss = tf.reduce_sum(object_mask * tf.square(
                object_confidence - object_confidence_hat) * best_confidence_mask) / self.batch_size

        return object_loss

    def calculate_no_object_confidence_loss(self, no_object_confidence_hat, no_object_confidence, no_object_mask,
                                            ignore_mask, scope='no_object_confidence_loss'):
        with tf.name_scope(scope):
            no_object_loss = tf.reduce_sum(no_object_mask * tf.square(
                no_object_confidence - no_object_confidence_hat) * ignore_mask) / self.batch_size

        return no_object_loss

    def calculate_xy_loss(self, label_object_mask, best_confidence_mask, txy_hat, txy, scope='xy_loss'):
        with tf.name_scope(scope):
            xy_loss = tf.reduce_sum(
                label_object_mask * best_confidence_mask * tf.square(txy - txy_hat)) / self.batch_size

        return xy_loss

    def calculate_wh_loss(self, label_object_mask, best_confidence_mask, twh_hat, twh, scope='wh_loss'):
        with tf.name_scope(scope):
            wh_loss = tf.reduce_sum(
                label_object_mask * best_confidence_mask * tf.square(twh - twh_hat)) / self.batch_size

        return wh_loss

    def calculate_classify_loss(self, object_mask, predicts_class, labels_class, scope='classify_loss'):
        with tf.name_scope(scope):
            class_loss = tf.reduce_sum(object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_class,
                                                                                             logits=predicts_class)) / self.batch_size
        return class_loss

    def calculate_iou(self, pred_box, label_box, scope="IOU"):
        with tf.name_scope(scope):
            with tf.name_scope("pred_box"):
                pred_box_xy = pred_box[..., 0:2]
                pred_box_wh = pred_box[..., 2:]
                pred_box_leftup = pred_box_xy - pred_box_wh / 2.0
                pred_box_rightdown = pred_box_xy + pred_box_wh / 2.0
                pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

            with tf.name_scope("label_box"):
                label_box_xy = label_box[..., 0:2]
                label_box_wh = label_box[..., 2:]
                label_box_leftup = label_box_xy - label_box_wh / 2.0
                label_box_rightdown = label_box_xy + label_box_wh / 2.0
                label_box_area = label_box_wh[..., 0] * label_box_wh[..., 1]

            with tf.name_scope("intersection"):
                intersection_leftup = tf.maximum(label_box_leftup, pred_box_leftup)
                intersection_rightdown = tf.minimum(label_box_rightdown, pred_box_rightdown)
                intersection_wh = tf.maximum(intersection_rightdown - intersection_leftup, 0.0)
                intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

            iou = tf.div(intersection_area, pred_box_area + label_box_area - intersection_area)
            return iou

    def scale_hat(self, feature_map, anchors, calculate_loss=False):
        # 预测值的特征图转向原始图
        anchor_tensor = tf.cast(anchors, tf.float32)
        anchor_tensor = tf.reshape(anchor_tensor, [1, 1, 1, self.num_anchors, 2])

        # 1.计算偏移量
        with tf.name_scope("create_grid_offset"):
            grid_shape = tf.shape(feature_map)[1:3]
            grid_y = tf.range(0, grid_shape[0])
            grid_x = tf.range(0, grid_shape[1])
            grid_y = tf.reshape(grid_y, [-1, 1, 1, 1])
            grid_x = tf.reshape(grid_x, [1, -1, 1, 1])
            grid_y = tf.tile(grid_y, [1, grid_shape[0], 1, 1])
            grid_x = tf.tile(grid_x, [grid_shape[1], 1, 1, 1])
            grid = tf.cast(tf.concat([grid_x, grid_y], axis=3), tf.float32)

        # 转换feature_map的形状，便于计算下面的置信度和解码操作
        feature_map = tf.reshape(feature_map, [-1, grid_shape[0], grid_shape[1], self.num_anchors, 5 + self.num_class])
        with tf.name_scope("scale_hat_activations"):
            bbox_confidence = tf.sigmoid(feature_map[..., 0:1], name="confidence")
            bbox_xy = tf.sigmoid(feature_map[..., 1:3], name="bbox_xy")
            bbox_wh = tf.exp(feature_map[..., 3:5], name="bbox_wh")
            bbox_class_probs = tf.sigmoid(feature_map[..., 5:], name="class_probs")

            # 将xywh进行归一化
            bbox_xy = (bbox_xy + grid) / tf.cast(grid_shape[0], tf.float32)
            bbox_wh = bbox_wh * anchor_tensor / tf.cast(self.image_size, tf.float32)

        if calculate_loss:
            return grid, feature_map, bbox_xy, bbox_wh

        return bbox_xy, bbox_wh, bbox_confidence, bbox_class_probs

    def total_loss(self, labels):

        grid_shape = [tf.cast(tf.shape(scale)[1:3], tf.float32) for scale in self.scales]
        # 1.对预测的每一层特征进行解码，解码到原图，便于计算下面的IOU
        for i in range(len(self.scales)):
            grid, predict, bbox_xy, bbox_wh = self.scale_hat(self.scales[i], self.anchors[self.anchor_mask[i]],
                                                             calculate_loss=True)

            pred_box = tf.concat([bbox_xy, bbox_wh], axis=-1)
            # 取出真实框的计算值，将xywh进行归一化
            # 有无对象的指示函数
            label_object_mask = tf.cast(labels[i][..., 0:1], tf.float32)
            label_object_mask = tf.expand_dims(label_object_mask, axis=3)
            # 五个候选框的指示函数
            label_object_mask = tf.tile(label_object_mask, [1, 1, 1, self.num_anchors, 1])

            label_no_object_mask = 1 - label_object_mask

            # 坐标的计算和归一化
            label_xy = tf.cast(labels[i][..., 1:3], tf.float32)
            label_xy = tf.expand_dims(label_xy, axis=3) / tf.cast(grid_shape[i], tf.float32)
            label_xy = tf.tile(label_xy, [1, 1, 1, self.num_anchors, 1])

            # 宽高的计算和归一化
            label_wh = tf.cast(labels[i][..., 3:5], tf.float32)
            label_wh = tf.expand_dims(label_wh, axis=3) / tf.cast(self.image_size, tf.float32)
            label_wh = tf.tile(label_wh, [1, 1, 1, self.num_anchors, 1])

            # 类别概率的计算
            label_class = tf.cast(labels[i][..., 5:], tf.float32)
            label_class = tf.expand_dims(label_class, axis=3)
            label_class = tf.tile(label_class, [1, 1, 1, self.num_anchors, 1])

            label_box = tf.concat([label_xy, label_wh], axis=-1)

            # 预测信息的转换xywh和类别概率
            confidence_hat = tf.sigmoid(predict[..., 0:1])
            txy_hat = tf.sigmoid(predict[..., 1:3])
            twh_hat = predict[..., 3:5]
            class_hat = predict[..., 5:]

            # 真实候选框的xywh和类别概率,需要进行编码
            anchor_tensor = tf.cast(self.anchors[self.anchor_mask[i]], tf.float32)
            anchor_tensor = tf.reshape(anchor_tensor, [1, 1, 1, self.num_anchors, 2])
            twh = tf.log(label_wh * tf.cast(self.image_size, tf.float32) / anchor_tensor)
            # 编码后只取出有对象的wh
            twh = tf.keras.backend.switch(label_object_mask, twh, tf.zeros_like(twh))
            txy = (label_xy * tf.cast(grid_shape[i], tf.float32) - grid) * label_object_mask

            iou = self.calculate_iou(pred_box, label_box)
            best_confidence = tf.reduce_max(confidence_hat, axis=-1, keepdims=True)
            best_confidence_mask = tf.cast(confidence_hat >= best_confidence, tf.float32)

            # iou小于阈值的视为无目标对象
            ignore_mask = tf.cast(iou < self.ignore_thresh, tf.float32)
            ignore_mask = tf.expand_dims(ignore_mask, axis=4)

            # 有无对象的置信度
            label_object_confidence = 1.0
            label_no_object_confidence = 0.0

            # 如果没有最好的iOU列为无对象损失中
            no_object_mask = (1 - best_confidence_mask) + label_no_object_mask

            # 计算有目标对象的损失
            object_confidence_loss = self.calculate_object_confidence_loss(confidence_hat, label_object_confidence,
                                                                           label_object_mask, best_confidence_mask,
                                                                           'object_confidence_loss_' + str(i))

            # 计算无目标对象的损失
            no_object_confidence_loss = self.calculate_no_object_confidence_loss(confidence_hat,
                                                                                 label_no_object_confidence,
                                                                                 no_object_mask, ignore_mask,
                                                                                 'no_object_confidence_loss_' + str(i))

            # 计算xywh的损失
            xy_loss = self.calculate_xy_loss(label_object_mask, best_confidence_mask, txy_hat, txy, 'xy_loss_' + str(i))
            wh_loss = self.calculate_wh_loss(label_object_mask, best_confidence_mask, twh_hat, twh, 'wh_loss_' + str(i))

            # 计算类别的损失
            class_loss = self.calculate_classify_loss(label_object_mask, class_hat, label_class,
                                                      'classify_loss_' + str(i))

            self.loss += self.coord_alpha * (xy_loss + wh_loss) + \
                         self.object_alpha * object_confidence_loss + \
                         self.no_object_alpha * no_object_confidence_loss + \
                         self.class_alpha * class_loss


def main():
    x_scale = 416 / 1069
    y_scale = 416 / 500
    scales_size = [13, 26, 52]
    """
    主函数，通过这里产生测试数据调用yolov3类进行损失函数的计算
    :return: 该图像的损失
    """
    # 1.加载图像制作标签
    img_tag = "010f658c-c912-4964-b796-90d3a9f34f19.jpg"
    img_corrd = "719_52_116_110_1;169_182_229_185_1;450_198_178_173_1;673_191_196_216_1"

    img_path = os.path.join(os.getcwd(), "data", "car", "image", "train_1w", img_tag)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (416, 416))
    image = tf.image.per_image_standardization(image)
    image = tf.cast(tf.expand_dims(image, 0), tf.float32)

    position_list = [corrd.split("_") for corrd in img_corrd.strip().split(";")]
    total_grid_attr = 6
    labels = []

    for scale_size in scales_size:
        maxtric = np.zeros([scale_size, scale_size, total_grid_attr])

        for position in position_list:
            if position[0] == "" or len(position) < 5:
                continue
            # xy是中心点坐标
            w = float(position[2]) * x_scale
            h = float(position[3]) * y_scale
            x = (float(position[0]) * x_scale + w / 2) / (416 / scale_size)
            y = (float(position[1]) * y_scale + h / 2) / (416 / scale_size)

            grid_x = int(x)
            grid_y = int(y)

            maxtric[grid_x, grid_y, 0] = 1.0
            maxtric[grid_x, grid_y, 1] = x
            maxtric[grid_x, grid_y, 2] = y
            maxtric[grid_x, grid_y, 3] = w
            maxtric[grid_x, grid_y, 4] = h
            maxtric[grid_x, grid_y, 4] = 1.0

        maxtric = np.expand_dims(maxtric, 0)
        labels.append(maxtric)
    net = darknet53(image, True)
    yolov3 = YOLOV3(net, labels, 1)
    # 2.初始化类

    # 3.返回图像的损失
    return yolov3.loss


if __name__ == "__main__":
    sess = tf.Session()
    loss = main()

    sess.run(tf.global_variables_initializer())
    print(sess.run(loss))
