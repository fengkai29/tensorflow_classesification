# -*- coding: utf-8 -*-

from skimage import io, transform
import glob
import os
from dataset.dataset import dataset,minibatches
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import MaxPool2D
from models import model, vgg

output = '/Users/fengkai/PycharmProjects/tensorflow_classification/output'
rawdata_root = '/Users/fengkai/PycharmProjects/tensorflow_classification/dataset/data/train'
all_pd = pd.read_csv("/Users/fengkai/PycharmProjects/tensorflow_classification/dataset/data/train.txt", sep=" ", header=None, names=['ImageName', 'label'])
train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43, stratify=all_pd['label'])

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 将所有的图片resize成100*100
width = 224
height = 224
image_size = (width, height, 3)
classes = 100


# 读取图片
# def read_img(path):
#     cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
#     imgs = []
#     labels = []
#     for idx, folder in enumerate(cate):
#         for im in glob.glob(folder + '/*.jpg'):
#             print('reading the images:%s' % (im))
#             img = io.imread(im)
#             img = transform.resize(img, (w, h))
#             imgs.append(img)
#             labels.append(idx)
#     return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# data, label = read_img(path)

# 打乱顺序
# num_example = data.shape[0]
# arr = np.arange(num_example)
# np.random.shuffle(arr)
# data = data[arr]
# label = label[arr]
#
# # 将所有数据分为训练集和验证集
# ratio = 0.8
# s = np.int(num_example * ratio)
# x_train = data[:s]
# y_train = label[:s]
# x_val = data[s:]
# y_val = label[s:]

# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, classes], name='y_')

# # 第一个卷积层（100——>50)
# conv1 = tf.layers.conv2d(
#     inputs=x,
#     filters=32,
#     kernel_size=[5, 5],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
# # 第二个卷积层(50->25)
# conv2 = tf.layers.conv2d(
#     inputs=pool1,
#     filters=64,
#     kernel_size=[5, 5],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
# # 第三个卷积层(25->12)
# conv3 = tf.layers.conv2d(
#     inputs=pool2,
#     filters=128,
#     kernel_size=[3, 3],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
#
# # 第四个卷积层(12->6)
# conv4 = tf.layers.conv2d(
#     inputs=pool3,
#     filters=128,
#     kernel_size=[3, 3],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
#
# re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])
#
# # 全连接层
# # dense1 = tf.layers.dense(inputs=re1,
# #                          units=1024,
# #                          activation=tf.nn.relu,
# #                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
# #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# # dense2 = tf.layers.dense(inputs=dense1,
# #                          units=512,
# #                          activation=tf.nn.relu,
# #                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
# #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# logits = tf.layers.dense(inputs=re1,
#                          units=100,
#                          activation=None,
#                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# # logits = tf.nn.softmax(logits)
# # ---------------------------网络结束---------------------------
# _ ,logits = model.Vgg(x,classes,isvgg19 = False).build()
logits = vgg.VGG16(x,classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits))
# loss += 0.0005 * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=1000,
                                           decay_rate=0.9)
add_global = global_step.assign_add(1)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
prediction = tf.to_int64(tf.argmax(logits, 1))
correct_prediction = tf.equal(prediction, tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 训练和测试数据，可将n_epoch设置更大一些
x_train, y_train = dataset(rawdata_root,train_pd,image_size,classes).images, dataset(rawdata_root,train_pd,image_size,classes).labels
x_val, y_val = dataset(rawdata_root,val_pd,image_size,classes).images, dataset(rawdata_root,val_pd,image_size,classes).labels
n_epoch = 10
batch_size = 8
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#saver = tf.tf.train.Saver()
for epoch in range(n_epoch):
    start_time = time.time()
    step = 0
    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        # print(x_train_a,y_train_a)
        _, err, ac = sess.run([add_global,train_op, loss, acc,learning_rate], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
        if step % 10 == 0:
            print(" train:epoch: %d || train loss: %f || train acc: %f" % (epoch,train_loss / n_batch,train_acc / n_batch))
            train_loss, train_acc, n_batch = 0, 0, 0

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print(" val:epoch: %d || train loss: %f || train acc: %f" % (epoch, val_loss / n_batch, val_acc / n_batch))
    tf.train.Saver().save(sess, save_path = os.path.join(output,"cnn-%d-%.2f" %(epoch,val_acc)),global_step=step)
sess.close()
