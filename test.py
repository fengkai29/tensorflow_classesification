import tensorflow as tf

# import numpy as np
# batch_size = 2
# a = [[1,2],[3,4],[4,5],[6,7]]
# indices = np.arange(len(a))
# np.random.shuffle(indices)
# print(indices)
# def a(a,batch_size):
#     for start_idx in range(0, len(a) - batch_size + 1, batch_size):
#         excerpt = indices[start_idx:start_idx + batch_size]
#         yield a[excerpt]
# print(a(a,batch_size))

random = tf.random.shuffle([True, False])
print(random[0])
import tensorflow as tf
a=tf.constant(2)
b=tf.constant(3)
x=tf.constant(4)
y=tf.constant(5)
z = tf.multiply(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
with tf.Session() as session:
    print(result.eval())