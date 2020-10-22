import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pickle

IMG_to_predict = '../data/imgs/test_watches/panerai/Luminor1713.jpeg'
closest_watches = []


def load_image(path):
    img = image.load_img(path, target_size=(224,224)) # target_size=model_VGG19.input_shape[1:3]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def get_closest_images(query_image_idx, num_results=5):
    distances = [distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]

    return idx_closest


def get_concatenated_images(indexes, thumb_height):
    thumbs = []

    with open('./data/WF_images.pickle', 'rb') as f:
        images = pickle.load(f)

    for idx in indexes:
        closest_watches.append(images[idx])
        img = image.load_img('.' + images[idx][2:])  # por el tema de las rutas relativas... desde el main es con ./
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


def load_feature_model(path):
    model = tf.keras.models.load_model(path)
    print('VGG19 loaded...')

    return model


def update_features(images):
    features = []
    for i, image_path in enumerate(images):
        img, x = load_image(image_path)
        feat = feat_extractor.predict(x)[0]
        features.append(feat)

    with open('../data/WF_features.pickle', 'wb') as f:
        pickle.dump(features, f)


# img, x = load_image(IMG_to_predict)
# print("data type: ", x.dtype)
# plt.imshow(img)
# plt.show()
#
# img, x = load_image(IMG_to_predict)
# feat = feat_extractor.predict(x)
# plt.figure(figsize=(16,4))
# plt.plot(feat[0])
# plt.show()


def check_models_on_sale():
    images_path = '../data/imgs/WF_panerai'
    image_extensions = ['.jpg', '.png', '.jpeg']
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
    with open('./data/WF_images.pickle', 'wb') as f:
        pickle.dump(images, f)
    print(f'Comparing to {len(images)} on sale.')

    return images


def make_suggestion(model, IMG_to_predict):
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    # images = check_models_on_sale()

    with open('./data/WF_features.pickle', 'rb') as f:
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
    plt.figure(figsize = (16,12))
    plt.imshow(results_image)
    plt.title("result images")
    plt.show()

    return closest_watches

