import numpy as np
import tifffile
from cellpose import models, core
import os
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
#from train_network import train_network
from u_net import UNet

def import_images(images_path,num_imgs=20):
    images = [np.squeeze(tifffile.imread(images_path + str(i) + '.tif')) for i in range(num_imgs)]
    return images

def get_cellpose_data(images_path, augment=False, learning_rate=0.01):
    num_imgs = 5

    #Import the images
    print('importing images')
    images = import_images(images_path,num_imgs=num_imgs)
    images = np.array(images)
    images = images
    print(images.shape)
    print(images.shape)
    print('imported images successfully')

    #Put the images through CellPose to get the cellprobs and masks
    print('getting model')
    model = models.CellposeModel(model_type='nuclei',gpu=core.use_gpu())
    print('got model')

    print('getting data')

    masks, flows, styles = model.eval(images,channels=[[0,0]],cellprob_threshold=False)

    print('got data')

    masks = np.array(masks)
    masks = np.where(masks>0,1,0)
    flows = np.array(flows)
    

    #cellprobs = [[0][i][:,:,2] for i in range(num_imgs)]
    cellprobs = np.array([flows[2][i] for i in range(num_imgs)])

    #each model needs to use the same X_train and y_train, we test them on the same data

    #First model
        #Takes original images in 256x256 crops
        #Takes the ground truth CellPose cellprobs
        #Trained

    #Second model
        #Takes the groundtruth CellPose cellprobs
        #Takes the groundtruth CellPose masks

    return images, masks, cellprobs


def train_model(images_path, augment=False, learning_rate=0.01):

    #Import the images
    print('importing images')
    images = import_images(images_path,num_imgs=5)
    images = np.array(images)
    print(images.shape)
    print('imported images successfully')

    #Put the images through CellPose to get the cellprobs and masks
    print('getting model')
    model = models.CellposeModel(model_type='nuclei',gpu=core.use_gpu())
    print('got model')

    print('getting data')
    masks, flows, styles = model.eval(images,channels=[[0,0]],cellprob_threshold=False,normalize=False)
    print('got data')

    masks = np.array(masks)
    flows = np.array(flows)

    #Split that data into X_train, X_test, y_train, y_test
    print(masks.shape)
    print(flows.shape)

    #each model needs to use the same X_train and y_train, we test them on the same data

    #First model
        #Takes original images in 256x256 crops
        #Takes the ground truth CellPose cellprobs
        #Trained

    #Second model
        #Takes the groundtruth CellPose cellprobs
        #Takes the groundtruth CellPose masks

    return model

def get_random_crops(images, masks, cellprobs):

    imgs = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]
    masks = [(mask-np.min(mask))/(np.max(mask)-np.min(mask)) for mask in masks] #they are already 0 and 1 no need to normalise
    cellprobs = [(cellprob-np.min(cellprob))/(np.max(cellprob)-np.min(cellprob)) for cellprob in cellprobs]

    #make random crops with them
    imgs_aug = []
    mks_aug = []
    cellprob_aug = []
    for i in range(len(imgs)):
        img = imgs[i]
        mask = masks[i]
        cellprob = cellprobs[i]
        for j in range(1000):
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
            img_cropped = np.array(img[y:y+crop_val, x:x+crop_val],dtype=np.float16)
            mask_cropped = mask[y:y+crop_val, x:x+crop_val]
            cellprob_cropped = cellprob[y:y+crop_val, x:x+crop_val]

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
            cellprob_cropped = np.expand_dims(cellprob_cropped,-1)

            img_cropped = np.moveaxis(img_cropped, -1, 0)
            mask_cropped = np.moveaxis(mask_cropped,-1,0)
            cellprob_cropped = np.moveaxis(cellprob_cropped,-1,0)

            imgs_aug.append(img_cropped)
            mks_aug.append(mask_cropped)
            cellprob_aug.append(cellprob_cropped)
    
    imgs_aug = torch.tensor(imgs_aug)
    mks_aug = torch.tensor(np.array(mks_aug))
    cellprob_aug = torch.tensor(np.array(cellprob_aug))

    return imgs_aug, mks_aug, cellprob_aug

def get_data_loaders(imgs_aug, mks_aug, cellprob_aug):
    random_state = 10

    X_train_img, X_test_img, y_train_cp, y_test_cp = train_test_split(imgs_aug, cellprob_aug, test_size=0.33, random_state=random_state)

    X_train_cp, X_test_cp, y_train_mks, y_test_mks = train_test_split(cellprob_aug, mks_aug, test_size=0.33, random_state=random_state)

    trainDS_img = [(X_train_img[i],y_train_cp[i]) for i in range(len(X_train_img))]
    testDS_img  = [(X_test_img[i],y_test_cp[i]) for i in range(len(X_test_img))]
    trainLoader_img = DataLoader(trainDS_img, shuffle=True,
	                    batch_size=5, pin_memory=True,
	                    num_workers=2)
    testLoader_img = DataLoader(testDS_img, shuffle=True,
                            batch_size=5, pin_memory=True,
                            num_workers=2)

    trainDS_cp = [(X_train_cp[i],y_train_mks[i]) for i in range(len(X_train_cp))]
    testDS_cp  = [(X_test_cp[i],y_test_mks[i]) for i in range(len(X_test_cp))]
    trainLoader_cp = DataLoader(trainDS_cp, shuffle=True,
	                    batch_size=5, pin_memory=True,
	                    num_workers=2)
    testLoader_cp = DataLoader(testDS_cp, shuffle=True,
                            batch_size=5, pin_memory=True,
                            num_workers=2)

    return trainLoader_img, testLoader_img, trainLoader_cp, testLoader_cp

#if __name__ == '__main__':

#    images_path = os.getcwd() + '/812_plate/'

    #model = train_model(images_path, augment=False, learning_rate=0.01)