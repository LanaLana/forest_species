import os
import cv2
import sys
import math
import scipy
import random
import rasterio
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import seed
from glob import glob
from rasterio.windows import Window

from keras.models import model_from_json
from keras import backend as K
from keras.layers import Conv2D
from keras import layers
from keras.models import Model
import tensorflow as tf

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    weights_list = np.zeros((len(keys)))
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        #score = mu*total/float(labels_dict[key])
        class_weight[key] = score if score > 1.0 else 1.0
        weights_list[sorted(keys).index(key)] = class_weight[key]
    return class_weight, weights_list

class Generator:
    def __init__(self, train_img_list, batch_size, val_img_list, class_0, class_1, num_channels):
        self.train_img_list = train_img_list
        self.val_img_list = val_img_list
        self.num_channels = num_channels
        self.num_classes = 2
        self.IMG_ROW = 64
        self.IMG_COL = 64
        self.batch_size = batch_size
        self.class_0 = class_0
        self.class_1 = class_1
        self.img_prob = np.ones(len(train_img_list)) / len(train_img_list)
            
    def get_img_mask_array(self, imgpath, age_flag = False):
        #class_0=['S'], class_1=['E']
        with rasterio.open(imgpath+'/'+'B02_10m_norm.tif') as src:
            size_x = src.width
            size_y = src.height

        rnd_x = random.randint(0,size_x -  self.IMG_ROW - 1)
        rnd_y = random.randint(0,size_y - self.IMG_COL - 1)
        window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
        #with rasterio.open(imgpath+ '_channel_01.tif') as src:
        #    img[:,:,0] = src.read(window=window)

        mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
        for cl_name in self.class_0:
            if '{}_05.tif'.format(cl_name) in os.listdir(imgpath):
                with rasterio.open(imgpath + '/{}_05.tif'.format(cl_name)) as src:
                    mask_0 += src.read(window=window).astype(np.float)
        mask_0 = mask_0 > 0.5

        mask_1 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
        for cl_name in self.class_1:
            if '{}_05.tif'.format(cl_name) in os.listdir(imgpath):
                with rasterio.open(imgpath + '/{}_05.tif'.format(cl_name)) as src:
                    mask_1 += src.read(window=window).astype(np.float)
        mask_1 = mask_1 > 0.5

        # remove black area
        while np.sum(mask_0+mask_1) < self.IMG_ROW*self.IMG_COL / 10: 
            #or np.count_nonzero(img[:,:,0]) < self.IMG_ROW*self.IMG_COL*3/5.:
            rnd_x = random.randrange(0,size_x -  self.IMG_ROW - 1) 
            rnd_y = random.randrange(0,size_y - self.IMG_COL - 1)
            window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
            #with rasterio.open(imgpath+ '_channel_01.tif') as src:
            #    img[:,:,0] = src.read(window=window)
            #parent_mask = np.zeros((1, self.IMG_ROW, self.IMG_COL))
            mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
            for cl_name in self.class_0:
                if '{}_05.tif'.format(cl_name) in os.listdir(imgpath):
                    with rasterio.open(imgpath + '/{}_05.tif'.format(cl_name)) as src:
                        mask_0 += src.read(window=window).astype(np.float)
            mask_0 = mask_0 > 0.5

            mask_1 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
            for cl_name in self.class_1:
                if '{}_05.tif'.format(cl_name) in os.listdir(imgpath):
                    with rasterio.open(imgpath + '/{}_05.tif'.format(cl_name)) as src:
                        mask_1 += src.read(window=window).astype(np.float)
            mask_1 = mask_1 > 0.5

        img = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.float)
        for i, ch in enumerate(['B02','B03','B04','B05','B06','B07','B08','B11','B12','B8A']):
            with rasterio.open(imgpath+'/'+'{}_10m_norm.tif'.format(ch)) as src:
                img[:,:,i] = src.read(window=window)
        
        img /= 255.

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # AGE
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if age_flag:
            channel_name = '_age.tif'
            with rasterio.open(imgpath + channel_name) as src:
                img[:,:,-1] = src.read(window=window).astype(np.float)
            img[:,:,-1] = (img[:,:,-1] / 100.).clip(0., 1.)

        mask = np.ones((self.IMG_ROW, self.IMG_COL, self.num_classes)) 
        mask[:,:,0] = mask_0  
        mask[:,:,1] = mask_1 

        return np.asarray(img), np.asarray(mask)  

    def train_gen(self):
        while(True):
            imgarr=[]
            maskarr=[]
            train_samples = np.random.choice(self.train_img_list, self.batch_size, p=self.img_prob)
            for i in range(self.batch_size):
                img,mask=self.get_img_mask_array(train_samples[i])
                imgarr.append(img)
                maskarr.append(mask)
            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[] 

    def val_gen(self):
        while(True):
            imgarr=[]
            maskarr=[]
            for i in range(self.batch_size):
                rnd_id=random.randint(0,len(self.val_img_list)-1)
                img,mask=self.get_img_mask_array(self.val_img_list[rnd_id])

                imgarr.append(img)
                maskarr.append(mask)

            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[]
            
    def set_prob(self):
        img_prob = np.zeros((len(self.train_img_list)))
        for i, img_path in enumerate(self.train_img_list):
            for cl in self.class_0+self.class_1:
                if cl+'_05.tif' in os.listdir(img_path):
                    img_prob[i] += np.sum(tiff.imread(img_path+'/'+cl+'_05.tif'))
        img_prob = img_prob/np.sum(img_prob)
        return img_prob

    def weighted_categorical_crossentropy(self, weights):
        def loss(target,output,from_logits=False):
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            non_zero_pixels = tf.reduce_sum(target, axis=-1)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1) \
                    * (self.IMG_ROW*self.IMG_COL*self.batch_size) / K.sum(non_zero_pixels)

        return loss
