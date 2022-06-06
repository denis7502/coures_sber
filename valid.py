
from pyexpat import model
import cv2
import torch
from torchvision import transforms
from torch import nn

def load_model(path):
    model = torch.load(path)#torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')

    model.eval()

    return model.to('cuda')

def prepr():
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return prep

def load_image(path, preprocces):
    img = cv2.imread(path)
    input_tensor = preprocces(img)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch.to('cuda')


def getDist(v1, v2):
    pdist = nn.PairwiseDistance(p=2)
    return pdist(v1,v2)


def main():
    model10 = load_model('pretrained-model_10.pt')
    model20 = load_model('pretrained-model_20.pt')
    model = load_model('pretrained-model.pt')
    prep = prepr()
    img1 = load_image(r'to\Friday 13 May 2022 01-19-06AM\21.84.jpg', prep)
    img2 = load_image(r'to\Wednesday 18 May 2022 01-53-06AM\21.84.jpg',prep)
    v1_m10 = model10(img1)
    v2_m10 = model10(img2)
    print(getDist(v1_m10, v2_m10), '10 epoch')
    
    v1_m20 = model20(img1)
    v2_m20 = model20(img2)
    print(getDist(v1_m20, v2_m20), '20 epoch')
    
    v1 = model(img1)
    v2 = model(img2)
    print(getDist(v1, v2), 'pretrain 10 epoch')



main()