from pathlib import Path
from tensorflow import keras
import cv2

def get_img():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()
    return train_images, train_labels

d = Path('../mnist')
d.mkdir(exist_ok=True)
imgs, labels = get_img()
for i in range(len(imgs)):
    p = d / '{}-{}.png'.format(i, labels[i])
    cv2.imwrite(str(p.resolve()), imgs[i])
    