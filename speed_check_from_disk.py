import tensorflow as tf
from tensorflow import keras
import time, random, sys
from pathlib import Path
import numpy as np
from pprint import pprint
import cv2
random.seed(10)

# 画像のパスとラベルのペアの numpy.ndarray を作成
def load_data():
    p = Path('../mnist')
    if not p.exists():
        print('path not found')
        sys.exit()
    pngs = p.glob('*png')
    #                      X-Y.png         Y.png          Y
    data = np.array([[str(i.resolve()), int(str(i).split('/')[-1].split('-')[-1].split('.')[0])] for i in pngs]).T

    return data[0], data[1]

def _parse_function(filename, labels):
    image_string = tf.read_file(filename)
    # decode_png の第二引数が 1 のとき グレースケールで読み込む
    image_decoded = tf.cast(tf.image.decode_png(image_string, 1), tf.float32)
    image_decoded = tf.div(image_decoded, 255.)
    image_decoded = tf.reshape(image_decoded, [28, 28, 1])
    onehot = tf.one_hot(labels, 10)
    return image_decoded, onehot

def next_element(img_files, labels, batch_size, num_para):
    tf_imgs = tf.constant(img_files)
    tf_labels = tf.constant(labels, tf.uint8)
    dataset = tf.data.Dataset.from_tensor_slices((tf_imgs, tf_labels))
    dataset = dataset.map(_parse_function, num_parallel_calls=num_para).shuffle(buffer_size=1000).batch(batch_size).prefetch(1).repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

class GetImage:
    def __init__(self, img_pathes, labels, batch_size):
        self.img_pathes = img_pathes
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

        imgs = []
        labels = []
        for i in batch:
            img = cv2.imread(self.img_pathes[i], 0).astype(np.float32)
            img_dev = (img / 255.).astype(np.float32)
            imgs.append(img_dev)
            labels.append(list(np.eye(10, dtype=np.uint8)[int(self.labels[i])]))
        return np.array(imgs).reshape((-1, 28, 28, 1)), np.array(labels)


def sessrun(l, sleep=0.):
    start = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(l[2]):
            if sleep != 0.:
                time.sleep(sleep)
            if l[0] == 0:
                sess.run(l[1])
            else:
                l[1].next_batch()
    return time.time() - start

if __name__ == '__main__':
    batch_size = 128
    imgs, labels = load_data()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    for sleep in [0., 0.2, 0.4]:
        for num_para in [2, 8]:
            # 三回計測
            print('\n')
            print('sleep: ', sleep, 'num_para: ', num_para)

            for x in range(3):
                print('num cycle: ', x)

                i = 7
                d = 2 ** i
                l = [[0, next_element(imgs, labels, batch_size, num_para), d], [1, GetImage(imgs, labels, batch_size), d]]
                print('dataset API:\t', sessrun(l[0], sleep))
                
                print('my Generator:\t', sessrun(l[1], sleep))
    
