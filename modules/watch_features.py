import pandas as pd


def process_watch_list_df(lst):

    """
    Returns suggested watches, names and images paths to create the selection box.
    """

    ids = []
    pams = []
    models = []
    pic_paths = []
    count = 1

    for i in lst:
        txt = i[24:-5].split('_')
        ids.append(txt[0])
        pams.append(txt[1])
        string = str(count) + '.- ' + txt[2]
        models.append(string)
        pic_paths.append((i))
        count += 1

    return pd.DataFrame({'id': ids, 'pam': pams, 'model': models, 'pic_path': pic_paths})


def load_watch_features(path, selected_id):
    """
    Returns selected watch features, hashtag and buying link
    """

    columns_name = {'id': 'ID', 'model': 'MODEL',
                    'pam': 'PAM', 'price': 'PRICE',
                    'retail_price': 'RETAIL PRICE',
                    'year': 'YEAR', 'box': 'BOX',
                    'papers': 'PAPERS', 'gender': 'GENDER',
                    'movement': 'MOVEMENT', 'case_size': 'CASE SIZE',
                    'case_material': 'CASE MATERIAL',
                    'bracelet_material': 'BRACELET MATERIAL',
                    'dial_type': 'DIAL TYPE', 'w_resistance': 'WATER RESISTANCE',
                    'link': 'LINK'}

    on_sale_df = pd.read_csv(path, usecols=columns_name.keys())
    on_sale_df.rename(columns=columns_name, inplace=True)

    selected_filter = on_sale_df['ID'] == selected_id
    watch_info = on_sale_df[selected_filter]
    tag = watch_info['PAM'].item()
    sale_link = watch_info['LINK'].item()
    watch_info = watch_info[['ID', 'MODEL', 'PAM', 'PRICE', 'RETAIL PRICE', 'YEAR', 'BOX',
                             'PAPERS', 'GENDER', 'MOVEMENT', 'CASE SIZE', 'CASE MATERIAL',
                             'BRACELET MATERIAL', 'DIAL TYPE', 'WATER RESISTANCE']].T

    return watch_info, tag, sale_link