import torch
from torchvision import transforms
import cv2
import psql
from datetime import datetime
import os

PATH_TO_DIR = r'D:\knower\jetson\course\frames'

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.eval()

    return model


def prepr():
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return prep


def read_img(path, preprocces):
    img = cv2.imread(path)
    input_tensor = preprocces(img)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


def use_model(inp, model):
    if torch.cuda.is_available():
        inp = inp.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(inp).detach().cpu().numpy()

    return str(output[0].tolist()).replace('[', '{').replace(']', '}')


def addToDB(output, path, bd):
    _, dest = path.split('frames')
    dst, date, pt = dest.split('\\')[1:]
    date = datetime.strptime(date, '%A %d %b %Y %I-%M-%S%p') 
    if dst == 'back':
        dest = False
    else:
        dest = True
    pt = float(pt.split('.j')[0])
    print(path)
    q = f"""INSERT INTO dataset (points, datecreate, vector, dest, path) VALUES ('{round(pt,2)}', '{date}', '{output}', '{dest}', '{path}')"""
    bd.query(q)


def main():
    bd = psql.DB()
    model = load_model()
    preprocces = prepr()
    os.chdir(PATH_TO_DIR)
    path_r = os.getcwd()
    root = os.listdir()
    for fold in root:
        if fold == 'to' or fold == 'back':
            folders = os.listdir(path_r + '\\' + fold)
            for folder in folders:
                imgs = os.listdir(path_r + '\\' + fold + '\\' + folder)
                for img in imgs:
                    path = path_r + '\\' + fold + '\\' + folder + '\\' + img
                    inp = read_img(path, preprocces)
                    out = use_model(inp, model)
                    addToDB(out, path, bd)



main()
