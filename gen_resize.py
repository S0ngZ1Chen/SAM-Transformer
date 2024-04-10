import sys
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import math

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * max(radius[0],radius[1]) + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    gaussian = cv2.resize(gaussian,dsize=(radius[0]*2+1,radius[1]*2+1),interpolation=cv2.INTER_LINEAR)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius[0]), min(width - x, radius[0] + 1)
    top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius[1] - top:radius[1] + bottom, radius[0] - left:radius[0] + right]
    '''
    plt.subplot(121)
    plt.imshow(gaussian)
    plt.subplot(122)
    plt.imshow(masked_gaussian)
    plt.show(block=True)
    '''
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

base_dir = sys.path[0]
save_size = (2048,2048)#w,h
save_prob_size = (512,512)
image_names = os.listdir(os.path.join(base_dir,'ori_images'))


for image_name in image_names:
    image = cv2.imread(os.path.join(base_dir,'ori_images',image_name))
    voc_path = os.path.join(base_dir,'voc_annotations',image_name.replace('.jpg','.xml'))
    tree = ET.parse(voc_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    bboxs = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(int((bbox.find('xmin').text))*save_prob_size[0]/width)
        ymin = int(int((bbox.find('ymin').text))*save_prob_size[1]/height)
        xmax = int(int((bbox.find('xmax').text))*save_prob_size[0]/width)
        ymax = int(int((bbox.find('ymax').text))*save_prob_size[1]/height)

        bboxs.append([xmin,ymin,xmax,ymax])

    #print(width,height)
    #print(bboxs)

    new_image = cv2.resize(image,dsize=save_size)
    cv2.imwrite(os.path.join(base_dir,'images',image_name.replace('.jpg','.png')),new_image)

    probmap = np.zeros((save_prob_size[1],save_prob_size[0]))
    for bbox in bboxs:
        x1,y1,x2,y2 = bbox
        h, w = y2 - y1, x2 - x1
        radius = (math.ceil(w/2),math.ceil(h/2))
        ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        probmap = draw_gaussian(probmap, ct_int, radius)
    cv2.imwrite(os.path.join(base_dir,'annotations',image_name.replace('.jpg','.png')),probmap*255)
    
    