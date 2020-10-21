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

IMG_to_predict = '../data/imgs/test_watches/panerai/Luminor1713.jpeg'

def main():

    panerai_or_no_model = pon.model_pan_or_not_load()
    model, feat_extractor = sg.load_feature_model()
    img, x = sg.load_image(IMG_to_predict)
    plt.imshow(img)

if __name__ == '__main__':
    main()