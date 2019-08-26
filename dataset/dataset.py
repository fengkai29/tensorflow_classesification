from skimage import io, transform
import os
import numpy as np
import tensorflow as tf

class dataset():
    def __init__(self,imgroot, anno_pd,image_size,classes = 100):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        labels = anno_pd['label'].tolist()
        self.classes = classes
        # labels = [i-1 for i in labels]
        # labels = tf.one_hot(labels,classes)
        new_label = []
        for i in range(len(labels)):
            label = np.zeros((self.classes))
            id = labels[i] - 1
            label[id] = 1
            new_label.append(label)
        self.image_size = image_size
        image = []
        for i in range(len(self.paths)):
            img_path = os.path.join(self.root_path, self.paths[i])
            img = io.imread(img_path)
            img = transform.resize(img, (self.image_size[0], self.image_size[1]))
            img = np.multiply(img, 1.0 / 255.0)
            image.append(img)
        self.images = np.asarray(image, np.float32)
        self.labels = np.asarray(new_label,np.int32)

    # 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]