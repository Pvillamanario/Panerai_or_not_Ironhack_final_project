import argparse
from predictions import pan_or_not_prediction as pon
from predictions import suggestions as sg
import matplotlib.pyplot as plt

# def argument_parser():
#     parser = argparse.ArgumentParser(description = 'Set chart type')
#     parser.add_argument("-b", "--bar", help="Produce a barplot", action="store_true")
#     parser.add_argument("-l", "--line", help="Produce a lineplot", action="store_true")
#     args = parser.parse_args()
#     return args

# def main(some_args):
#     data = mac.acquire()
#     filtered = mwr.wrangle(data, year)
#     results = man.analyze(filtered)
#     fig = mre.plotting_function(results, title, arguments)
#     mre.save_viz(fig, title)
#     print('========================= Pipeline is complete. You may find the results in the folder ./data/results =========================')
#
# if __name__ == '__main__':
#     year = int(input('Enter the year: '))
#     title = 'Top 10 Manufacturers by Fuel Efficiency ' + str(year)
#     arguments = argument_parser()
#     main(arguments)

model_pan_or_not_path = './models/model_Inception_pan_or_not.h5'
VGG19_feat_extractor_path = './models/VGG19_ft_ext.h5'
feat_extractor_path = './models/feature_extractor.h5'

# IMG_to_predict = './data/imgs/test_watches/not_panerai/Not_Panerai_Cartier_39.jpeg'
IMG_to_predict = './data/imgs/test_watches/panerai/Luminor1713.jpeg'
# IMG_to_predict = './data/imgs/test_watches/panerai/Radiomir580.jpeg'
# IMG_to_predict = './data/imgs/test_watches/panerai/Submersible_C24_370.jpeg'


def main():

    model_pan_or_not = pon.model_pan_or_not_load(model_pan_or_not_path)
    model_VGG19 = sg.load_feature_model(VGG19_feat_extractor_path)
    img, x = sg.load_image(IMG_to_predict)
    plt.imshow(img)
    plt.show()
    is_panerai = pon.pan_or_not_prediction(IMG_to_predict, model_pan_or_not)

    if is_panerai != 'Panerai':
        print("Sorry, it doesn't seem to be a Panerai watch")

    else:
        print('It seems to be a Panerai watch!')
        similar_watches = sg.make_suggestion(model_VGG19, IMG_to_predict)
        print(similar_watches)



if __name__ == '__main__':
    main()