import caffe

import numpy as np
from PIL import Image

import random
#from data_agu import DataAugmentation

class layer_6_channel(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        #split_f  = '{}/{}.txt'.format(self.voc_dir,
        #        self.split)
        split_f  = '{}{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.txt_len = len(self.indices)-1
        self.idx = 0
        self.img_h = 227
        self.img_w = 227

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        tmp = self.indices[self.idx].split()
        data1 = self.load_image("../images/"+tmp[0])
        data2 = self.load_image("../images/"+tmp[0])
        self.data = self.concatenation_img(data1, data2)
        self.label = self.load_label(tmp[1])
        #print "==========================="
        #print self.idx
        #print self.data.shape, self.label.shape
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        #print top
        #self.idx = self.idx + 1
        #self.idx = self.idx % self.txt_len
        '''if self.idx % 100 == 0:
            self.split_num += 1
            self.split_num = self.split_num % 5
            print ("*-------------------------------*")
            print self.txt_len
            print ("*-------------------------------*")
            print("######################")'''
        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def concatenation_img(self, img1, img2):
        img = np.dstack((img1, img2))
        img = img[:,:,::-1]
        img = img - self.mean
        img = img.transpose((2,0,1))
        return img

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        - data_agumetation(add gaussian noise and color jittering) (add by Peter Huang)
        """
        im = Image.open(idx)  # Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        #img = im.resize((480, 480),Image.ANTIALIAS)  
        image_in = im.resize((227, 227),Image.ANTIALIAS)
        #print "Training phase."
        #sharp = DataAugmentation.randomColor(image_in)
        #noise = DataAugmentation.randomGaussian(sharp)
        #image_in = noise
        in_ = np.array(image_in, dtype=np.float32)
        #in_1 = np.array(image_in, dtype=np.float32)
        #in_2 = np.array(image_in, dtype=np.float32)
        #in_ = np.dstack((in_1,in_2))
        #in_ = in_[:,:,::-1]
        #in_ -= self.mean
        #in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, label):
        return np.array(label, dtype=np.uint8)
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        '''#print idx
        im = Image.open(idx) #Image.open('{}/SegmentationClass/{}.jpg'.format(self.voc_dir, idx))
        #img = im.resize((480, 480),Image.ANTIALIAS)
        #object_list = ['folgers','crayola','kleenex','viva','vanish','milo','swissmiss','cocacola','raisins','mm','andes','pocky','kotex','macadamia','stax','kellogg','hunts','3m','heineken','libava']
        object_list = ['buoy', 'dock', 'totem']
        ### temporary change ###
        try:
            for obj_idx in range(0, len(object_list)):
                if idx.find(object_list[obj_idx])!=-1 :
                    im_mask = np.asarray(im)
                    area = im_mask[:,:] == 255
                    seg = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                    seg[area==True] = obj_idx + 1
                    break
        except:
            print("Load Error!!!")
        label = np.array(seg, dtype=np.uint8)
        ### temporary change ###
        ##=label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]'''