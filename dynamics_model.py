import torch
import os
import random
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import ModuleList
from torch.nn import MaxPool2d
from torch.nn import ConvTranspose2d
import torch.nn.functional as F
from torch.optim import Adam
import time
from torch.autograd import Variable
from cellpose import models
from cellpose import core

#import data

def get_images_masks(images_path,masks_path,num_imgs=20):
    images = []
    masks = []
    for i in range(num_imgs):
        images.append(np.squeeze(tifffile.imread(images_path + str(i) + '.tif')))
        masks.append(np.squeeze(tifffile.imread(masks_path + str(i) + '.tif')))

    return images, masks

#Function to prepare data for U-Net
#Input: images and masks
#Output: training and testing data
def get_data(images,masks,channel=0,flows=False):
    imgs = [image[:,:,channel] for image in images] #only keep first channel
    imgs = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in imgs] #normalise between 1 and 0 with min-max values

    #binarise masks
    if flows == False:
        mks = [(mask > 0) * 1 for mask in masks] #binarise masks
    else:
        model = models.CellposeModel(model_type='nuclei',gpu=core.use_gpu())
        masks, flows, styles = model.eval(images,channels=[[0,0]],cellprob_threshold=False,normalize=False)
        all_flows = np.array([flows[0][2]]) #works because for now we are only using 1 image!!!!

        #I think we should normalise these flows
        mks = np.array([(mks-np.min(mks))/(np.max(mks)-np.min(mks)) for mks in all_flows])
        #mks = all_flows

        print('mks 1:',np.unique(mks[0]))
        print(mks.shape)

    #make random crops with them
    imgs_aug = []
    mks_aug = []
    for i in range(len(imgs)):
        img = imgs[i]
        mask = mks[i]
        for j in range(100):
            #crop_width = random.randint(5,256)
            #crop_height = random.randint(5,256)
            #crop_val = random.randint(5,256)
            crop_val = 256
            assert img.shape[0] >= crop_val
            assert img.shape[1] >= crop_val
            assert img.shape[0] == mask.shape[0]
            assert img.shape[1] == mask.shape[1]
            x = random.randint(0, img.shape[1] - crop_val)
            y = random.randint(0, img.shape[0] - crop_val)
            img_cropped = img[y:y+crop_val, x:x+crop_val]
            mask_cropped = mask[y:y+crop_val, x:x+crop_val]

            #Filters out the masks that have only background
            if len(np.unique(mask_cropped)) == 1:
                j -= 1
                continue
            
            #only allow samples where the amount of cells takes over 0.4 of the total image
            unique, counts = np.unique(mask_cropped, return_counts=True)
            #while len(counts) < 2 or (counts[1] / (counts[0]+counts[1])) < 0.4:
            
            #Not padding anymore because all the images are of the same size (128x128)
            #img_cropped = padding(img_cropped,256,256)
            #mask_cropped = padding(mask_cropped,256,256)

            img_cropped = np.expand_dims(img_cropped,-1)
            mask_cropped = np.expand_dims(mask_cropped,-1)

            img_cropped = np.moveaxis(img_cropped, -1, 0)
            mask_cropped = np.moveaxis(mask_cropped,-1,0)

            imgs_aug.append(img_cropped)
            mks_aug.append(mask_cropped)

    #make them torches
    imgs = torch.tensor(np.array(imgs_aug))
    mks = torch.tensor(np.array(mks_aug))

    print('Printing shapes')
    print(imgs.shape)
    print(mks.shape)

    #return with training and testing split
    X_train, X_test, y_train, y_test = train_test_split(imgs, mks, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

class Block(Module):
	
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) #CONV->RELU->CONV


class Encoder(Module):

	def __init__(self, channels=(1, 16, 32, 64)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)

	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs

class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2)for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
    # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
            # return the final decoder output
        return x
		
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures

class UNet(Module):
    def __init__(self, encChannels=(1, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=1, retainDim=True, outSize=(256, 256)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        x = x.type(torch.float32)
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
            encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        #print(map.shape)
        #print('1st',map.shape)
        #if map.shape[1] < 128 and map.shape[2] < 128 and self.retainDim:
            #print('interpolating')
        #    map = F.interpolate(map, self.outSize)
            #map = F.pad(map, pad=(31, 31, 31, 31), mode='constant', value=0)
        #print('2nd',map.shape)
        # return the segmentation map
        #map = torch.sigmoid(map)
        #map = torch.where(map>0.5, 1.0, 0)
        #map = map == 1.0
        #print(map)
        return map


if __name__ == '__main__':

    #try to make prediction without training model
    #look at prediction at epoch 0

    #Checking if PyTorch MPS (for M1) is activated
    #print('PyTorch has MPS activated:',torch.backends.mps.is_built())
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    images_path = os.getcwd() + '/812_plate/'
    masks_path = os.getcwd() + '/812_plate_masks/'

    images, masks = get_images_masks(images_path, masks_path,num_imgs=1)

    X_train, X_test, y_train, y_test = get_data(images,masks,flows=True)

    print(X_train.shape)

    model_path = os.getcwd() + "\saved_model\modelone30epochs"
    
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda:0")
    model.eval()

    testDS = [(X_test[i],y_test[i]) for i in range(len(X_test))]
    testLoader = DataLoader(testDS, shuffle=True,
                            batch_size=5, pin_memory=True,
                            num_workers=2)

    for (x, y) in testLoader:
        # send the input to the device
        x,y=x.type(torch.float32),y.type(torch.float32)
        #(x, y) = (x.to("mps"), y.to("mps"))
        (x,y) = (x.to("cuda:0"), y.to("cuda:0"))
        # make the predictions and calculate the validation loss
        
        pred = model(x)
        break

        #totalTestLoss += lossFunc(pred, y)
    
    #pred = model(testLoader)


    plt.imshow(pred)
    plt.show()
