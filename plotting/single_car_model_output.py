# Input Dimension: 1024x1024
# Input Dimension: 512x512

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
from tqdm import tqdm
from keras.models import load_model

WIDTH, HEIGHT, BATCH_SIZE = 1024, 1024, 2
# WIDTH, HEIGHT, BATCH_SIZE = 512, 512, 8
ORIG_WIDTH = 1918
ORIG_HEIGHT = 1280

def run_length_encode(mask):
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


df_test = pd.read_csv('single_car_model_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for _id in ids_test:
    names.append('{}.jpg'.format(_id))

    
from keras.losses import binary_crossentropy
import keras.backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

model = load_model(
    filepath='weights/model_weights_1024.hdf5',
    # filepath='weights/model_weights_512.hdf5',
    custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef}
)

graph = tf.get_default_graph()
q_size = 10


def data_loader(q, ):
    for start in range(0, len(ids_test), BATCH_SIZE):
        x_batch = []
        
        end = min(start + BATCH_SIZE, len(ids_test))
        ids_test_batch = ids_test[start:end]
        
        for id in ids_test_batch.values:
            img = cv2.imread('single_car_model_Train/{}.jpg'.format(id))
            img = cv2.resize(img, (WIDTH, HEIGHT))
            
            x_batch.append(img)
        
        x_batch = np.array(x_batch, np.float32) / 255.0
        q.put(x_batch)


rles = []

def predictor(q, ):
    for _ in tqdm(range(0, len(ids_test), BATCH_SIZE)):
        x_batch = q.get()
        
        with graph.as_default():
            preds = model.predict_on_batch(x_batch)
        
        preds = np.squeeze(preds, axis=3)
        
        for pred in preds:
            prob = cv2.resize(pred, (ORIG_WIDTH, ORIG_HEIGHT))
            mask = prob > 0.5
            rle = run_length_encode(mask)
            rles.append(rle)

            
q = queue.Queue(maxsize=q_size)

t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))

t1.start()
t2.start()

t1.join()
t2.join()

df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('rle_encoded/1024.csv.gz', index=False, compression='gzip')
# df.to_csv('rle_encoded/512.csv.gz', index=False, compression='gzip')