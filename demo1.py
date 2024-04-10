import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from model import Sam_Head
from train import load_checkpoint
import cv2
import math

if __name__ == '__main__':
    base_dir = sys.path[0]
    data_dir = os.path.join(base_dir, 'data')
    split = 'test'

    model_path = os.path.join(base_dir,'run','train','2024-04-09-00-12-01','best_mae_trainval_0.pth.tar')
    image_paths = os.path.join(data_dir, 'images')
    mask_paths = os.path.join(data_dir, 'annotations')
    embedding_paths = os.path.join(data_dir, 'embeddings')
    image_names_path = os.path.join(data_dir, split+'.txt')

    with open(image_names_path) as f:
        image_names = []
        lines = f.readlines()#读取全部内容
        for line in lines:
            image_name = line.replace('\n','')
            image_names.append(image_name)

    model = Sam_Head().cuda()
    model,_ = load_checkpoint(model,model_path)


    for image_name in tqdm(image_names):
        img = plt.imread(os.path.join(image_paths,image_name+'.png'))
        embedding = np.squeeze(np.load(os.path.join(embedding_paths,image_name+'.npy')),0)
        mask = np.expand_dims(plt.imread(os.path.join(mask_paths,image_name+'.png')),2)
        #print(img.shape,embedding.shape,mask.shape)
        with torch.no_grad():
            outputs = model(torch.from_numpy(embedding).unsqueeze(0).cuda())

        ax = plt.subplot(231)
        plt.axis('off')
        plt.imshow(img)
        ax2 = plt.subplot(232)
        plt.axis('off')
        plt.imshow(mask)
        plt.subplot(233)
        plt.axis('off')
        plt.imshow(np.transpose(np.max(np.expand_dims(embedding,0),axis=1),(1,2,0)))
        plt.subplot(234)
        plt.axis('off')
        plt.imshow(np.transpose(outputs.cpu()[0].numpy(),(1,2,0)))
        plt.show(block=True)