import sys
import os
import numpy as np
import matplotlib.pyplot as plt
base_dir = sys.path[0]
images = os.listdir(os.path.join(base_dir,'data','images'))
images = ['CDY_2015.png']
for image in images:
    image_path = os.path.join(base_dir,'data','images',image)
    img = plt.imread(image_path)
    embedding_path = os.path.join(base_dir,'data','embeddings',image.replace('.png','.npy'))
    #embedding = np.load(embedding_path)
    annotation_path = os.path.join(base_dir,'data','annotations',image)
    ann = plt.imread(annotation_path)
    #print(img.shape,embedding.shape,ann.shape)

    '''
    for i in range(embedding.shape[1]):
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(np.transpose(embedding[:,i,:,:],(1,2,0)))
        plt.show(block=True)
    '''
    
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(ann)
    plt.subplot(133)
    #plt.imshow(np.transpose(np.max(embedding,axis=1),(1,2,0)))
    plt.show(block=True)

