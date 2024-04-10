from torch.utils.data import Dataset
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

class SH_Dataset(Dataset):
    def __init__(self, data_dir, split = 'trainval', train = True):
        self.split = split
        self.image_paths = os.path.join(data_dir, 'images')
        self.mask_paths = os.path.join(data_dir, 'annotations')
        self.embedding_paths = os.path.join(data_dir, 'embeddings')
        self.image_names_path = os.path.join(data_dir, split+'.txt')
        self.train = train

        with open(self.image_names_path) as f:
            self.names = []
            lines = f.readlines()#读取全部内容
            for line in lines:
                name = line.replace('\n','')
                self.names.append(name)

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        if self.train:
            image, embedding, mask, name  = self.load_augmented_img(index)
        else:
            image, embedding, mask, name  = self.load_img(index)
        #print(embedding.shape,mask.shape,name)
        '''
        base_dir = sys.path[0]
        cv2.imwrite(os.path.join(base_dir,'run','vis_data_augment',name+'.png'), np.transpose(image,(1,2,0)))
        cv2.imwrite(os.path.join(base_dir,'run','vis_data_augment',name+'_embedding.png'), cv2.resize(np.transpose(np.max(np.expand_dims(embedding,0),axis=1),(1,2,0))*255,(mask.shape[1],mask.shape[2])))
        cv2.imwrite(os.path.join(base_dir,'run','vis_data_augment',name+'_mask.png'), np.transpose(mask,(1,2,0))*255)
        '''
        '''
        plt.subplot(131)
        plt.imshow(np.transpose(image,(1,2,0)))
        plt.subplot(132)
        plt.imshow(np.transpose(np.max(np.expand_dims(embedding,0),axis=1),(1,2,0)))
        plt.subplot(133)
        plt.imshow(np.transpose(mask,(1,2,0)))
        plt.show(block=True)
        '''
        return embedding, mask, name

    def load_img(self, index):
        name = self.names[index]
        image = cv2.imread(os.path.join(self.image_paths,name+'.png'))
        embedding = np.squeeze(np.load(os.path.join(self.embedding_paths,name+'.npy')),0)
        mask = np.expand_dims(plt.imread(os.path.join(self.mask_paths,name+'.png')),0)

        image = cv2.resize(image,(mask.shape[1],mask.shape[2]))
        image = np.transpose(image,(2,0,1))

        return image, embedding, mask, name
    
    def load_augmented_img(self, index):
        data_list = []
        image, embedding, mask, name  = self.load_img(index)

        if random.random() > 0.5:
            eh = embedding.shape[1]
            ew = embedding.shape[2]
            mh = mask.shape[1]
            mw = mask.shape[2]
            s = int(mh/eh)
            ch = random.randint(0,eh)
            cw = random.randint(0,ew)
            new_image = np.zeros_like(image)
            new_embedding = np.zeros_like(embedding)
            new_mask = np.zeros_like(mask)
            if random.random() > 0.5:#在原图中进行mosaic数据增广
                th = random.randint(0,eh-ch)
                tw = random.randint(0,ew-cw)
                c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+ch)*s,tw*s:(tw+cw)*s],embedding[:,th:th+ch,tw:tw+cw],mask[:,th*s:(th+ch)*s,tw*s:(tw+cw)*s])
                new_image[:,0:ch*s,0:cw*s] = c_image
                new_embedding[:,0:ch,0:cw] = c_embedding
                new_mask[:,0:ch*s,0:cw*s] = c_mask
                if ch!=eh:
                    th = random.randint(0,ch)
                    tw = random.randint(0,ew-cw)
                    c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+eh-ch)*s,tw*s:(tw+cw)*s],embedding[:,th:th+eh-ch,tw:tw+cw],mask[:,th*s:(th+eh-ch)*s,tw*s:(tw+cw)*s])
                    new_image[:,ch*s:,0:cw*s] = c_image
                    new_embedding[:,ch:,0:cw] = c_embedding
                    new_mask[:,ch*s:,0:cw*s] = c_mask
                if cw!=ew:
                    th = random.randint(0,eh-ch)
                    tw = random.randint(0,cw)
                    c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+ch)*s,tw*s:(tw+ew-cw)*s],embedding[:,th:th+ch,tw:tw+ew-cw],mask[:,th*s:(th+ch)*s,tw*s:(tw+ew-cw)*s])
                    new_image[:,0:ch*s,cw*s:] = c_image
                    new_embedding[:,0:ch,cw:] = c_embedding
                    new_mask[:,0:ch*s,cw*s:] = c_mask
                if ch!=eh and cw!=ew:
                    th = random.randint(0,ch)
                    tw = random.randint(0,cw)
                    c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+eh-ch)*s,tw*s:(tw+ew-cw)*s],embedding[:,th:th+eh-ch,tw:tw+ew-cw],mask[:,th*s:(th+eh-ch)*s,tw*s:(tw+ew-cw)*s])
                    new_image[:,ch*s:,cw*s:] = c_image
                    new_embedding[:,ch:,cw:] = c_embedding
                    new_mask[:,ch*s:,cw*s:] = c_mask
                return new_image, new_embedding, new_mask, name
            else:#在所有图像中进行mosaic数据增广
                th = random.randint(0,eh-ch)
                tw = random.randint(0,ew-cw)
                c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+ch)*s,tw*s:(tw+cw)*s],embedding[:,th:th+ch,tw:tw+cw],mask[:,th*s:(th+ch)*s,tw*s:(tw+cw)*s])
                new_image[:,0:ch*s,0:cw*s] = c_image
                new_embedding[:,0:ch,0:cw] = c_embedding
                new_mask[:,0:ch*s,0:cw*s] = c_mask
                if ch!=eh:
                    th = random.randint(0,ch)
                    tw = random.randint(0,ew-cw)
                    new_index = random.randint(0,len(self.names)-1)
                    image, embedding, mask, _  = self.load_img(new_index)
                    c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+eh-ch)*s,tw*s:(tw+cw)*s],embedding[:,th:th+eh-ch,tw:tw+cw],mask[:,th*s:(th+eh-ch)*s,tw*s:(tw+cw)*s])
                    new_image[:,ch*s:,0:cw*s] = c_image
                    new_embedding[:,ch:,0:cw] = c_embedding
                    new_mask[:,ch*s:,0:cw*s] = c_mask
                if cw!=ew:
                    th = random.randint(0,eh-ch)
                    tw = random.randint(0,cw)
                    new_index = random.randint(0,len(self.names)-1)
                    image, embedding, mask, _  = self.load_img(new_index)
                    c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+ch)*s,tw*s:(tw+ew-cw)*s],embedding[:,th:th+ch,tw:tw+ew-cw],mask[:,th*s:(th+ch)*s,tw*s:(tw+ew-cw)*s])
                    new_image[:,0:ch*s,cw*s:] = c_image
                    new_embedding[:,0:ch,cw:] = c_embedding
                    new_mask[:,0:ch*s,cw*s:] = c_mask
                if ch!=eh and cw!=ew:
                    th = random.randint(0,ch)
                    tw = random.randint(0,cw)
                    new_index = random.randint(0,len(self.names)-1)
                    image, embedding, mask, _  = self.load_img(new_index)
                    c_image, c_embedding, c_mask = batch_augment(image[:,th*s:(th+eh-ch)*s,tw*s:(tw+ew-cw)*s],embedding[:,th:th+eh-ch,tw:tw+ew-cw],mask[:,th*s:(th+eh-ch)*s,tw*s:(tw+ew-cw)*s])
                    new_image[:,ch*s:,cw*s:] = c_image
                    new_embedding[:,ch:,cw:] = c_embedding
                    new_mask[:,ch*s:,cw*s:] = c_mask
                return new_image, new_embedding, new_mask, name
        else:#只进行上下与左右颠倒操作
            image, embedding, mask = batch_augment(image, embedding, mask)
            return image, embedding, mask, name
        
def batch_augment(image, embedding, mask, flipud = 0.5, fliplr = 0.5):
    image = np.transpose(image,(1,2,0))
    embedding = np.transpose(embedding,(1,2,0))
    mask = np.transpose(mask,(1,2,0))
    if random.random() < flipud:
        image = np.flipud(image).copy()
        embedding = np.flipud(embedding).copy()
        mask = np.flipud(mask).copy()
    if random.random() < fliplr:
        image = np.fliplr(image).copy()
        embedding = np.fliplr(embedding).copy()
        mask = np.fliplr(mask).copy()
    image = np.transpose(image,(2,0,1))
    embedding = np.transpose(embedding,(2,0,1))
    mask = np.transpose(mask,(2,0,1))
    for i in range(embedding.shape[0]):
        f = random.uniform(0.9,1.1)
        embedding[i,:,:] = embedding[i,:,:]*f

    return image, embedding, mask

if __name__ == '__main__':
    base_dir = sys.path[0]
    data_dir = os.path.join(base_dir, 'data')
    dataset = SH_Dataset(data_dir, split = 'trainval', train = True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, (embeddings, masks, name) in enumerate(data_loader):
        if i == 10:
            break