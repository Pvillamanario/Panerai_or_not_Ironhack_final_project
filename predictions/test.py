import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pickle

closest_watches = []


def load_image(path):
    img = image.load_img(path, target_size=(224,224)) # target_size=model_VGG19.input_shape[1:3]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_closest_images(query_image_idx, num_results=5):
    distances = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]

    return idx_closest


def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        closest_watches.append(images[idx])
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


IMG_to_predict = '../data/imgs/test_watches/panerai/Luminor1713.jpeg'

model = tf.keras.models.load_model('../models/VGG19_ft_ext.h5')
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

# images = check_models_on_sale()

with open('../data/WF_images.pickle', 'rb') as f:
    images = pickle.load(f)

with open('../data/WF_features.pickle', 'rb') as f:
    features = pickle.load(f)

print('features loaded')

features = np.array(features)
pca = PCA(n_components=276)
pca.fit(features)
pca_features = pca.transform(features)

# load image and extract features
new_image, x = load_image(IMG_to_predict)
new_features = feat_extractor.predict(x)

# project it into pca space
new_pca_features = pca.transform(new_features)[0]

# calculate its distance to all the other images pca feature vectors
distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
results_image = get_concatenated_images(idx_closest, 200)

# display the results
plt.figure(figsize=(4, 4))
plt.imshow(new_image)
plt.title("query image")
plt.show()

# display the resulting images
plt.figure(figsize=(16, 12))
plt.imshow(results_image)
plt.title("result images")
plt.show()

print(closest_watches)
