import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from skimage import transform
from scipy.spatial import distance
import pickle
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import os


# Loads InceptionV3 model (Panerai or not).
def model_pan_load(path):
    print('Panerai model loaded...')
    return tf.keras.models.load_model(path)


# Checks if VGG19 is downloaded; if not, dowload it:
def check_vgg19(path):
    if os.path.isfile(path):
        print('VGG19 already dowloaded')
        pass
    else:
        model = tf.keras.applications.VGG19(weights='imagenet', include_top=True)
        model.save(path)


# If VGG19 model exists, loads it.
def load_feature_model(path):
    print('VGG19 loaded...')
    return tf.keras.models.load_model(path)


# Given an image, returns if a watch is Panerai or not.
def pan_prediction(uploaded_file, model):

    img = Image.open(uploaded_file)
    img = np.array(img).astype('float32')
    img = transform.resize(img, (180, 180, 3))
    img = np.expand_dims(img, axis=0)

    result = model.predict(img, batch_size=1)
    return result


# PCA dimensions reduction
def pca_reduction(features):
    features = np.array(features)
    pca = PCA(n_components=250)
    pca.fit(features)
    pca_features = pca.transform(features)

    return pca_features, pca


# Given a Panerai image, returns 6 similar watches.
def model_suggestion(uploaded_file, model, image_list, feature_list):

    img = Image.open(uploaded_file)
    img = np.array(img).astype('float32')
    img = transform.resize(img, (224, 224, 3))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    with open(image_list, 'rb') as f:
        images = pickle.load(f)
    # st.write('Checking watches on sale')

    # Removing decision layer
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    with open(feature_list, 'rb') as f:
        features = pickle.load(f)
    # st.write('Looking for similar watches')

    # Feature PCA reduction
    pca_features, pca = pca_reduction(features)

    # Feature extraction
    new_features = feat_extractor.predict(x)

    # Project it into pca space
    new_pca_features = pca.transform(new_features)[0]

    # Calculate its distance to all the other images pca feature vectors

    distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:6]  # grab first 3

    # Closest watches paths
    closest_watches = []

    for idx in idx_closest:
        closest_watches.append(images[idx][1:])

    print(closest_watches)

    return closest_watches
