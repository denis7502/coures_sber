import os

STEP = 1.68
FULL_DIST = 184.8

def rename(path,  reverse=False):
    imgs = os.listdir(path)
    if not reverse:
        cur_name = STEP
    else:
        cur_name = FULL_DIST
    for img in imgs:
        try:
            os.rename(path + '\\' + img, path + '\\' + str(round(cur_name,2)) + '.jpg')
        except FileExistsError:
            pass
        if not reverse:
            cur_name += STEP
        else:
            cur_name -= STEP


path_r = os.getcwd()
root = os.listdir()
for fold in root:
    if fold == 'to':
        folders = os.listdir(path_r + '\\' + fold)
        for folder in folders:
            rename(path_r + '\\' + fold + '\\' + folder)
    elif fold == 'back':
        folders = os.listdir(path_r + '\\' + fold)
        for folder in folders:
            rename(path_r + '\\' + fold + '\\' + folder, reverse=True)
