import tensorflow as tf
from tensorflow import keras
import time, random, sys
from pathlib import Path
import numpy as np
from pprint import pprint
random.seed(10)

def load_data():
    p = Path('data')
    if not p.exists():
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
    onehot = tf.one_hot(labels, 10)
    return imgs, onehot

def next_element(img_files, labels, batch_size):
    tf_imgs = tf.constant(img_files)
    tf_labels = tf.constant(labels, tf.uint8)
    dataset = tf.data.Dataset.from_tensor_slices((tf_imgs, tf_labels))
    dataset = dataset.map(_parse_function, num_parallel_calls=2).shuffle(buffer_size=(len(imgs))).batch(batch_size).prefetch(1).repeat()
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

        one_hot = lambda x: np.eye(10)[x]
        
        for i in batch:
            imgs.append(cv2.imread(self.img_pathes[i], 0))
            labels.append(cv2.imread(self.labels[i]))

        yield np.array(self.imgs[batch]), np.array(one_hot(self.labels[batch]))



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
            for i in range(100):
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
                if steps % 10 == 0:
                    print(steps)
                    print(time.time() - s_tmp)
                    s_tmp = time.time()
                if steps % 100 == 0:
                    print('total time:', time.time() - s, '\n')
                    s = time.time()
                    break

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(100):
#         i += 1

