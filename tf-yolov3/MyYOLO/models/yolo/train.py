import os
import math
import time
import numpy as np
import tensorflow as tf

from models.yolo import net
from models.yolo.YOLOv3 import YOLOv3
from models.yolo.util import config as cfg
from models.yolo.dataset.DataSet import DataSet


def train(data_dir, batch_size, net_name,
          epochs, start_epoch, start_iter, change_train_data_epoch,
          learning_rate, decay_rate, decay_steps,
          val_rate, save_rate,
          checkpoint_dir, log_dir,
          training):

    print('==> Get train and test data...')
    dataloader = DataSet(data_dir, batch_size, training)
    train_1w_batch = dataloader.train_1w_loader()
    train_b_batch = dataloader.train_b_loader()
    train_data_size_list = np.array([dataloader.nbr_train_1w, dataloader.nbr_train_b])
    print('==> Finished!')

    print('==> Create YOLOv3')
    print('--- use ', net_name)
    inputs_x = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3])
    model = net.__dict__[net_name](inputs_x, training)

    total_grid_cell_attr = 5 + cfg.NUM_CLASS
    inputs_y = [tf.placeholder(tf.float32, [None, cfg.SCALES[0], cfg.SCALES[0], total_grid_cell_attr]),
                tf.placeholder(tf.float32, [None, cfg.SCALES[1], cfg.SCALES[1], total_grid_cell_attr]),
                tf.placeholder(tf.float32, [None, cfg.SCALES[2], cfg.SCALES[2], total_grid_cell_attr])]
    yolo_v3 = YOLOv3(model, inputs_y, batch_size=batch_size, is_training=training)
    print('==> Finished!')

    print('==> Get each scale total loss')
    loss = yolo_v3.loss
    print('==> Finished!')

    print('==> Create optimizer')
    print('--- epochs = %d' % epochs)
    print('--- train_data_size = ', train_data_size_list)
    print('--- learning_rate = %f' % learning_rate)

    print('--- update learning_rate: ')
    print('--- \tlearning_rate = learning_rate * decay_rate^(global_step / decay_step)')
    print('--- decay_rate = %f' % decay_rate)

    total_step_list = [change_train_data_epoch * np.ceil(train_data_size_list[0] / batch_size), 
                        (epochs - change_train_data_epoch) * np.ceil(train_data_size_list[1] / batch_size)]
    print('--- total_step = ', total_step_list)

    print('--- start_epochs = %d' % start_epoch)

    train_iter_max_list = np.ceil(train_data_size_list / batch_size)
    print('--- train iter_max = ', train_iter_max_list)

    global_step = tf.Variable(0, trainable=False)
    print('change train data epoch = %d' % (change_train_data_epoch))  # [0,0,1],[0,1,0],[1,0,0]/

    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               decay_steps, decay_rate, staircase=False)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    print('==> Finished!')
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), flush_secs=60)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options.allow_growth = True
    config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=config) as sess:
        print('==> Load checkpoing')
        if len(os.listdir(checkpoint_dir)) >= 4:
            print('--> Restoring checkpoint from: ' + checkpoint_dir)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                sess.run(tf.local_variables_initializer())


            print('==> Load finished!')
        else:
            print('==> No checkpoint, start training new model')
            print('==> Init global variables')

            sess.run(tf.group(tf.global_variables_initializer()))
            train_writer.add_graph(sess.graph)

            print('==> Init finished!')

        print('==> Training start')

        print('--- save_rate = %d' % save_rate)

        print('--- val_rate = %d' % val_rate)

        save_path = os.path.join(checkpoint_dir)
        start_time = time.time()

        step = sess.run(global_step)
        # print("global_step:",global_step)
        temp_epoch = step//train_iter_max_list[0]
        # print("temp_epoch:",temp_epoch)
        temp_iter = step%train_iter_max_list[0]
        if temp_epoch<change_train_data_epoch:
            epoch= temp_epoch
            iter = temp_iter
        else:
            epoch=change_train_data_epoch+temp_iter//train_iter_max_list[1]
            iter = temp_iter%train_iter_max_list[1]

        print('--- epoch = %d' % epoch)
        print('--- iter = %d' % iter)

        train_iter_max = train_iter_max_list[0] if epoch < change_train_data_epoch else train_iter_max_list[1]


        # i = 0
        val_iter = 0
        total_step = sum(total_step_list)

        while epoch < epochs:
            while iter < train_iter_max:

                # ================== train ==================
                if epoch > change_train_data_epoch:
                    print("正在转换数据，使用未增强数据....")
                    batch = next(train_b_batch)
                else:
                    batch = next(train_1w_batch)


                feed_dict = {
                    inputs_x: batch[0],
                    inputs_y[0]: batch[1][0],
                    inputs_y[1]: batch[1][1],
                    inputs_y[2]: batch[1][2]
                }

                _, total_loss = sess.run([train_op, loss], feed_dict=feed_dict)
                # mask = sess.run(yolo_v3.best_confidence_mask_test,feed_dict=feed_dict)
                # print("置信度的mask:{}".format(mask.shape))
                # print(np.unique(mask))
                total_loss = total_loss*(batch_size/batch[0].shape[0])
                print("batch_size:",batch_size)
                print("实际批次的样本数：",batch[0].shape[0])
                print(batch_size/batch[0].shape[0])
                eta = remain_time(start_time, total_step, iter)
                print('--- Epoch {}, Iter {}, ETA {:.2f}m, loss {:.3f}'.format(epoch, iter, eta, total_loss))
                # ================== train ==================

                # ================== val ==================
                #if (step + 1) % val_rate == 0:
                    #print('==> Val test start')
        
                    #val_step = 0
                    #val_log_path = os.path.join(log_dir, 'val' + str((step + 1) // val_rate))
                    
                    #if not os.path.isdir(val_log_path):
                        #os.makedirs(val_log_path)
                    #val_writer = tf.summary.FileWriter(val_log_path, flush_secs=60)

                    #if epoch >= change_train_data_epoch:
                        #val_batch = dataloader.val_b_loader()
                        #print('--- val data size: ', dataloader.nbr_val_b)
                    #else:
                        #val_batch = dataloader.val_1w_loader()
                        #print('--- val data size: ', dataloader.nbr_val_1w)

                    #for batch in val_batch:
                        #feed_dict = {
                            #inputs_x: batch[0],
                            #inputs_y[0]: batch[1][0],
                            #inputs_y[1]: batch[1][1],
                            #inputs_y[2]: batch[1][2]
                        #}
                        #val_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                        #total_val_loss += val_loss
                        #val_writer.add_summary(summary_str, val_step)
                        #val_step += 1

                    #total_val_loss /= val_step

                    #print('-- Val loss {:.3f}'.format(total_val_loss), end=' ')

                    #if total_val_loss < best_loss:
                        #tmp = best_loss
                        #best_loss = total_val_loss
                        #print('Better then {:.3f}'.format(tmp))

                        #model_name = save_path + os.sep + 'yolov3.ckpt' + '-epoch_' + str(epoch) + '_' + \
                                     #str(iter) + '-bestloss_' + '%.3f' % best_loss
                        #saver.save(sess, save_path=model_name, global_step=step)
                        #print('--- save checkpoint best_loss: %.3f' % best_loss)
                    #else:
                        #print('Not better than {:.3f}'.format(best_loss))

                    #total_val_loss = 0
                # ================== val ==================

                if (iter + 1) % save_rate == 0:
                    model_name = save_path + os.sep + 'yolov3.ckpt' + '-epoch_' + str(epoch) + '_' +str(iter) + '-loss_' + str('%.3f' % total_loss)
                    saver.save(sess, save_path=model_name, global_step=global_step)
                    print('--- save checkpoint loss: %.3f' % total_loss)

                start_time = time.time()

                # i += 1
                iter += 1
                val_iter += 1

            with open("./logs/log.txt", mode="a") as file:
                file.write(str(epoch) + " " + str(total_loss) + "\n")
            iter = 0
            epoch += 1

        model_name = save_path + os.sep + 'yolov3.ckpt' + '-epoch_' + str(epoch) + '_' + \
                    str(iter) + '-loss_' + '%.3f' % total_loss
        saver.save(sess, save_path=model_name, global_step=global_step)
        print('--- save checkpoint loss: %.3f' % total_loss)

        print('==> Training Finished!')

def remain_time(start_time, total_step, step):
    end_time = time.time()
    during_time = end_time - start_time
    eta_s = (total_step - step) * during_time
    return eta_s / 60


# if __name__ == "__main__":
