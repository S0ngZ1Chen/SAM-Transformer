import sys
import os
import numpy as np
import matplotlib.pyplot as plt
base_dir = sys.path[0]
split = 'trainval'
data_dir = os.path.join(base_dir, 'data')
image_names_path = os.path.join(data_dir, split+'.txt')
with open(image_names_path) as f:
    names = []
    lines = f.readlines()#读取全部内容
    for line in lines:
        name = line.replace('\n','')
        names.append(name)

for name in names:
    embedding_path = os.path.join(base_dir,'run','vis_data_augment',name+'_embedding.npy')
    embedding = np.load(embedding_path)
    mask_path = os.path.join(base_dir,'run','vis_data_augment',name+'_mask.npy')
    mask = np.load(mask_path)
    
    plt.subplot(121)
    plt.imshow(np.transpose(mask,(1,2,0)))
    plt.subplot(122)
    plt.imshow(np.transpose(np.max(np.expand_dims(embedding,0),axis=1),(1,2,0)))
    plt.show(block=True)