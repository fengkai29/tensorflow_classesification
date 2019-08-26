import os
import numpy
import tensorflow as tf

output = '/home/kai.feng/tensorflow_classification/output/vgg16'
# 读取数据
def read_meta_info(root_dir, meta_file):
    meta_datas = open(meta_file).readlines()
    img_paths = []
    labels = []

    for meta_data in meta_datas:
        img_paths.append([os.path.join(root_dir, meta_data.split()[0]).replace('\\', '/')])
        labels.append(int(meta_data.split()[1]) - 1)

    return (numpy.array(img_paths), numpy.array(labels))


def _parse_train_function(img_paths, labels):
    images = []
    for index in range(img_paths.shape[0]):
        image = tf.read_file(img_paths[index])
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32) / 255.0
        images.append(tf.reshape(tf.image.resize_images(image, [224, 224]), [224, 224, 3]))

    images = tf.concat(images, axis=0)
    # images = tf.image.resize_images(images, [224,224])
    random = tf.random.shuffle([True, False])
    images = tf.cond(random[0], lambda: tf.image.flip_left_right(images), lambda: images)

    return images, tf.one_hot(tf.cast(labels, tf.int32), 100)


def create_data_pipeline(root_dir, meta_file):
    dataset = tf.data.Dataset.from_tensor_slices(read_meta_info(root_dir, meta_file))
    dataset = dataset.map(_parse_train_function)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size=batchsize)
    dataset = dataset.repeat()

    return dataset

#vgg网络定义
VGG_MEAN = [103.939, 116.779, 123.68]


def conv(x, d_out, name):
    d_in = x.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        # truncated_normal
        # kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
        # bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),trainable=True, name='bias')

        # xavier
        kernel = tf.get_variable(scope + 'weights', shape=[3, 3, d_in, d_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),trainable=True, name='bias')

        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv, name=scope)
        # activation = tf.nn.relu(conv + bias, name=scope)

        return activation


def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name)

    return activation


def fc(x, n_out, name):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:

        # truncated_normal
        # weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
        # bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), trainable=True, name='bias')

        # xavier
        weight = tf.get_variable(scope + 'weights', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), trainable=True, name='bias')

        if name == 'fc8':
            activation = tf.matmul(x, weight) + bias
        else:
            activation = tf.nn.relu_layer(x, weight, bias, name=name)

        return activation


def VGG16(images, _dropout, n_cls):
    rgb_scaled = images * 255.0
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])

    conv1_1 = conv(images, 64, 'conv1_1')
    conv1_2 = conv(conv1_1, 64, 'conv1_2')
    pool1 = maxpool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 128, 'conv2_1')
    conv2_2 = conv(conv2_1, 128, 'conv2_2')
    pool2 = maxpool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 256, 'conv3_1')
    conv3_2 = conv(conv3_1, 256, 'conv3_2')
    conv3_3 = conv(conv3_2, 256, 'conv3_3')
    pool3 = maxpool(conv3_3, 'pool3')

    conv4_1 = conv(pool3, 512, 'conv4_1')
    conv4_2 = conv(conv4_1, 512, 'conv4_2')
    conv4_3 = conv(conv4_2, 512, 'conv4_3')
    pool4 = maxpool(conv4_3, 'pool4')

    conv5_1 = conv(pool4, 512, 'conv5_1')
    conv5_2 = conv(conv5_1, 512, 'conv5_2')
    conv5_3 = conv(conv5_2, 512, 'conv5_3')
    pool5 = maxpool(conv5_3, 'pool5')

    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
    fc6 = fc(flatten, 1000, 'fc6')
    dropout1 = tf.nn.dropout(fc6, _dropout)

    fc7 = fc(dropout1, 500, 'fc7')
    dropout2 = tf.nn.dropout(fc7, _dropout)

    fc8 = fc(dropout2, n_cls, 'fc8')

    return fc8

#训练代码
root_dir = '/home/kai.feng/pytorch_classification/dataset/train_improve/data_all'
meta_file = '/home/kai.feng/pytorch_classification/dataset/train_improve/train.txt'
classes = 2
batchsize = 32

dataset = create_data_pipeline(root_dir, meta_file,batchsize)
train_iterator = dataset.make_one_shot_iterator()

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
next_element = iterator.get_next()

picture = tf.identity(next_element[0], name='picture')
dropout = tf.placeholder(tf.float32, name='dropout')
label = tf.identity(next_element[1], name='label')

predict = VGG16(picture, dropout, classes)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=label))
loss += 0.0005 * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
train_ops = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1)), tf.float32))
# config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session() as sess:
    step = 0
    training_handle = sess.run(train_iterator.string_handle())
    sess.run(tf.global_variables_initializer())

    while True:
        step = step + 1
        _ = sess.run([train_ops], feed_dict={handle: training_handle, dropout: 1.0})

        if step % 10 == 0:
            _loss, _accuracy = sess.run([loss, accuracy], feed_dict={handle: training_handle, dropout: 1.0})
            print("[TRN] Step:{0} loss:{1} accuray:{2}".format(step, _loss, _accuracy * 100))
        if step % 200 == 0:
            _loss, _accuracy = sess.run([loss, accuracy], feed_dict={handle: training_handle, dropout: 1.0})
            print("saving model")
            tf.train.Saver().save(sess,
                                  save_path=os.path.join(output, "vgg16-%d-%.2f.ckpt" % (step, _accuracy)),
                                  global_step=step)

        if step % 50000 == 0:
            break