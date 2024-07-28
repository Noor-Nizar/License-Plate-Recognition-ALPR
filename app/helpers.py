def get_label_paths(img_paths):
    ''' Get label paths from image paths '''
    return [x.replace('images', 'labels').replace('.jpg', '.txt') for x in img_paths]

def get_LP(paths):
    GTS = []
    for path in paths:
        with open(path) as file:
            GTS.append(file.readline().split()[-1])

    return GTS