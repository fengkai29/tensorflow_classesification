import tensorflow as tf
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


def VGG16(images, n_cls):
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
    dropout1 = tf.nn.dropout(fc6, 0.5)

    fc7 = fc(dropout1, 500, 'fc7')
    dropout2 = tf.nn.dropout(fc7, 0.5)

    fc8 = fc(dropout2, n_cls, 'fc8')

    return fc8