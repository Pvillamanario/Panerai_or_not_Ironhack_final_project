closest_watches = ['./data/imgs/WF_panerai/122826_PAM00183_Radiomir Manual.jpeg',
                   './data/imgs/WF_panerai/161440_PAM00609_Radiomir 8 days.jpeg',
                   './data/imgs/WF_panerai/149729_PAM00183_Radiomir Manual.jpeg',
                   './data/imgs/WF_panerai/167065_PAM00609_Radiomir 8 days.jpeg',
                   './data/imgs/WF_panerai/130917_PAM00183_Radiomir Manual.jpeg']


def process_watch_list(lst):

    id = []
    pam = []
    model = []

    for i in lst:
        txt = i[24:-5].split('_')
        id.append(txt[0])
        pam.append(txt[1])
        model.append((txt[2]))

    return id, pam, model
