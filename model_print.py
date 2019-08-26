import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = '/Users/fengkai/PycharmProjects/tensorflow_classification/output/vgg16-9-0.96.ckpt-1270' # 保存的ckpt文件名，不一定是这个
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
a = []
for key in var_to_shape_map:
    # print("tensor_name: ", key)
    a.append(key)
print(len(a))

include = ['fc8/bias', 'fc7/weights', 'conv5_1/weights', 'conv5_2/weights', 'conv4_2/weights', 'conv3_3/weights', 'fc7/bias', 'conv1_1/weights', 'conv3_2/weights', 'conv1_2/weights', 'conv3_1/weights', 'conv2_1/weights', 'beta1_power', 'fc6/bias', 'conv4_1/weights', 'fc8/weights', 'beta2_power', 'conv5_3/weights', 'fc6/weights', 'conv4_3/weights', 'conv2_2/weights']
print(len(include))