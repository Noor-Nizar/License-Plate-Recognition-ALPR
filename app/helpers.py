def get_label_paths(img_paths):
    ''' Get label paths from image paths '''
    return [f"data/raw/{file_name.split('/')[-1].split('.')[0]}.txt" for file_name in img_paths]

def get_LP(paths):
    GTS = []
    for path in paths:
        with open(path) as file:
            GTS.append(file.readline().split()[-1])

    return GTS