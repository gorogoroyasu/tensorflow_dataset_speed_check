import tensorflow as tf
from tensorflow import keras
import numpy as np
import time, random

random.seed(10)

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()
    return train_images, train_labels


def _parse_function(imgs, labels):
  onehot = tf.one_hot(labels, 10)
  imgs = tf.div(imgs, 255.)
  return imgs, onehot

def next_element(imgs, labels, batch_size):
    imgs = tf.constant(imgs, tf.float32)
    tf_labels = tf.constant(labels, tf.uint8)
    dataset = tf.data.Dataset.from_tensor_slices((tf_imgs, tf_labels))
    dataset = dataset.map(_parse_function, num_parallel_calls=2).shuffle(buffer_size=(len(imgs))).batch(batch_size).prefetch(1).repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

class GetImage:
    def __init__(self, imgs, labels, batch_size):
        self.imgs = imgs
        self.labels = labels
        self.order = []
        self.batch_size = batch_size
    
    def randomalize(self):
        x = [i for i in range(len(labels))]
        random.shuffle(x)
        if self.order == []:
            self.order = x
        else:
            self.order.extend(x)

    def next_batch(self):
        if len(self.order) < self.batch_size:
            self.randomalize()
        batch = self.order[:self.batch_size]
        self.order = self.order[self.batch_size:]

        one_hot = lambda x: np.eye(10)[x]

        yield self.imgs[batch], one_hot(self.labels[batch])



if __name__ == '__main__':
    batch_size = 128
    imgs, labels = load_data()
    next_elm = next_element(imgs, labels, batch_size)
    get_image = GetImage(imgs, labels, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        s = time.time()
        s_tmp = time.time()
        for t in [0, 1]:
            for i in range(1000):
                if t == 0:
                    if i == 0:
                        print('dataset')
                    sess.run(next_elm)
                else:
                    if i == 0:
                        print('my generator')
                    get_image.next_batch()
                steps = i + 1
                # time.sleep(0.05)
                if steps % 100 == 0:
                    print(steps)
                    print(time.time() - s_tmp)
                    s_tmp = time.time()
                if steps % 1000 == 0:
                    print('total time:', time.time() - s, '\n')
                    s = time.time()
                    break

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(100):
#         i += 1

