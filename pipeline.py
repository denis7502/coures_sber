import psql
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import random
from torch import nn
import numpy as np

def getRow(num, bd):
    q=f'select * from dataset where id = {num};'
    res = bd.query(q)
    id, pt, date, vec, dest, path = res[0]
    vec = torch.Tensor(vec)
    return id, pt, date, vec, dest, path

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.train()

    return model.to('cuda')

def prepr():
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return prep

def update(bd, vector, num):
    vector = str(vector[0].tolist()).replace('[', '{').replace(']', '}')
    q = f"""UPDATE dataset SET vector = '{vector}' where id = {num};"""
    bd.query(q)

def trainLoop( bd, prepr, model, count, epochs=100):
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(margin=1.68)
    running_loss = []
    paths = bd.query('select path from dataset')
    for epoch in range(epochs):
        for idx in tqdm(range(1, count+1), desc="Row"):
            id, pt, date, vec, dest, path = getRow(idx, bd)
            ids_a = bd.query(f'select id from dataset where points = {pt}')
            a_id = random.choice(ids_a)
            id_a, pt_a, date_a, vec_a, dest_a, path_a = getRow(a_id[0], bd)
            ids_n = bd.query(f'select id from dataset where points <> {pt}')
            n_id = random.choice(ids_n)
            id_n, pt_n, date_n, vec_n, dest_n, path_n = getRow(n_id[0], bd)
            
            #print(n_id)
            pos = load_image(path, prepr)
            neg = load_image(path_n, prepr)
            aun = load_image(path_a, prepr)

            optimizer.zero_grad()

            v_p = model(pos)
            v_n = model(neg)
            v_a = model(aun)

            loss = criterion(v_a, v_p, v_n)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            new_v = model(pos)
            update(bd, new_v, idx)
        
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
    torch.save(model, 'pretrained-model.pt')




def load_image(path, preprocces):
    img = cv2.imread(path)
    input_tensor = preprocces(img)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch.to('cuda')

def main():
    bd = psql.DB()
    prep = prepr()
    model = load_model()
    count = bd.query('select count(*) from dataset;')[0][0]

    trainLoop(bd, prep, model, count, 10)



main()