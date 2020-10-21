import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image




# image folder
img_to_predict_path = '../test'


def model_pan_or_not_load():

    base_learning_rate = 0.0001


    model_pan_or_not = tf.keras.models.load_model('../models/model_Inception_pan_or_not.h5')

    # model_pan_or_not.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    #                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #                           metrics=['accuracy'])

    print('panerai_model_loaded')
    return model_pan_or_not


def model_pan_or_not_prediction(img_path):

    result = []
    # load all images into a list
    images = []
    for img in os.listdir(img_path):
        print(img)
        img = os.path.join(img_path, img)
        img = image.load_img(img, target_size=(180, 180))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
    #     images.append(img)

    # stack up images list to pass for prediction
    #     images = np.vstack(images)
        classes = model_pan_or_not.predict(img, batch_size=1)

        if classes >= 0:
            result = 'Panerai'
        else:
            result = 'Not_Panerai'

    return result


# model_pan_or_not = model_pan_or_not_load(model_pan_or_not_path)
# prediction = model_pan_or_not_prediction(img_to_predict_path)
# print(prediction)