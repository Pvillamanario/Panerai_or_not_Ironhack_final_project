import tensorflow as tf


def get_vgg19():
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=True)
    model.save('../data/models/VGG19_ft_ext.h5')
    print('VGG19 saved at ./models')