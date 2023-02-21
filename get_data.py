import numpy as np
import tifffile
import torch
from cellpose import models, core
import random

def get_images_masks(images_path,masks_path,num_imgs=20):
    images = []
    masks = []
    for i in range(num_imgs):
        images.append(np.squeeze(tifffile.imread(images_path + str(i) + '.tif')))
        masks.append(np.squeeze(tifffile.imread(masks_path + str(i) + '.tif')))

    return images, masks


def padding(array, xx, yy):
    h = array.shape[0]
    w = array.shape[1]
    a = (xx - h) // 2
    aa = xx - a - h
    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

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
        for j in range(10):
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