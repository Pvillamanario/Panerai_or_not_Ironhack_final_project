import pandas as pd

choosen_watch = '../data/imgs/WF_panerai/149944_PAM00507_Luminor Submersible.jpeg'

choosen_watch = choosen_watch[24:-5].split('_')

df_features = pd.read_csv('../data/WF_panerai_features.csv')

print((df_features[df_features['id'] == choosen_watch[0]]).T)

