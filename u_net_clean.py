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
import cv2

#Import images
#Import masks
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
def get_data(images,masks,channel=0):
    imgs = [image[:,:,channel] for image in images] #only keep first channel
    imgs = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in imgs] #normalise between 1 and 0 with min-max values

    #binarise masks
    mks = [(mask > 0) * 1 for mask in masks] #binarise masks

    #make random crops with them
    imgs_aug = []
    mks_aug = []
    for i in range(len(imgs)):
        img = imgs[i]
        mask = mks[i]
        for j in range(100):
            crop_width = random.randint(5,250)
            crop_height = random.randint(5,250)
            assert img.shape[0] >= crop_height
            assert img.shape[1] >= crop_width
            assert img.shape[0] == mask.shape[0]
            assert img.shape[1] == mask.shape[1]
            x = random.randint(0, img.shape[1] - crop_width)
            y = random.randint(0, img.shape[0] - crop_height)
            img_cropped = img[y:y+crop_height, x:x+crop_width]
            mask_cropped = mask[y:y+crop_height, x:x+crop_width]

            #if len(np.unique(mask_cropped)) == 1:
            #    j -= 1
            #    continue
            
            #only allow samples where the amount of cells takes over 0.4 of the total image
            unique, counts = np.unique(mask_cropped, return_counts=True)
            if counts[0]/

            img_cropped = padding(img_cropped,250,250)
            mask_cropped = padding(mask_cropped,250,250)

            img_cropped = np.expand_dims(img_cropped,-1)
            mask_cropped = np.expand_dims(mask_cropped,-1)
            img_cropped = np.moveaxis(img_cropped, -1, 0)
            mask_cropped = np.moveaxis(mask_cropped,-1,0)

            imgs_aug.append(img_cropped)
            mks_aug.append(mask_cropped)

    #make them torches
    imgs = torch.tensor(np.array(imgs_aug))
    mks = torch.tensor(np.array(mks_aug))

    #return with training and testing split
    X_train, X_test, y_train, y_test = train_test_split(imgs, mks, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

#Create training and testing data
#The data is mix of randonmly cropped images
                        

class Block(Module):
	
	def __init__(self, inChannels, outChannels):
		super().__init__()
		self.conv1 = Conv2d(inChannels, outChannels, 4)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 4)

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
    def __init__(self, encChannels=(1, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=1, retainDim=True, outSize=(250,  250)):
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
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        # return the segmentation map
        map = torch.sigmoid(map)
        map = torch.where(map>0.5, 1.0, 0)
        #map = map == 1.0
        #print(map)
        return map

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class BinSegLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BinSegLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        pos_pixels = targets.sum()
        total_pixels = inputs.numel()

        pos_weight = total_pixels / (2 * pos_pixels + total_pixels)
        #print(~targets)
        #print(-1-targets)
        #print(inputs)
        #print(1-~targets)
        pos_loss = -targets * torch.log(inputs) * pos_weight
        neg_loss = -(1-targets) * torch.log(~inputs)

        return (pos_loss + neg_loss).mean()

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss

class FocalLoss_X(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss_X, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        
        BCE = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.7, beta = 0.3, epsilon=1e-6, gamma=3):
        inputs = torch.where(inputs>0.5, 1.0, 0)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives 
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        Tversky = (TP + epsilon)/(TP + alpha*FP + beta*FN + epsilon)
        tversky_loss = 1 - Tversky
        loss = torch.pow(tversky_loss, 0.75)
        print(loss)
        return loss

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

def train_network(trainLoader, testLoader,NUM_EPOCHS=10):
    DEVICE = "mps"
    unet = UNet().to(DEVICE)
    #lossFunc = DiceLoss()
    #lossFunc = BinSegLoss()
    #lossFunc = FocalLoss(gamma=2)
    #lossFunc = FocalLoss_X()
    #lossFunc = FocalTverskyLoss()
    lossFunc = IoULoss()
    opt = Adam(unet.parameters(), lr=0.00001)
    trainSteps = len(trainDS)
    testSteps = len(testDS)
    H = {"train_loss": [], "test_loss": []}

    print("[INFO] training the network...")
    device = torch.device(DEVICE)

    startTime = time.time()
    for e in range(NUM_EPOCHS):
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        #print('going through trainLoader')
        training_loader_iter = iter(trainLoader)
        #print('going throug loop')
        for i in range(len(trainLoader)):
            #print(str(i) + '/' + str(len(trainLoader)))
            # send the input to the device
            x, y = next(iter(training_loader_iter))
            x,y=x.type(torch.float32),y.type(torch.float32)
            (x, y) = (x.to("mps"), y.to("mps"))
            # perform a forward pass and calculate the training loss
            x = x.float()
            #print(x.shape)
            pred = unet(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss = Variable(loss, requires_grad = True)
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        #print('going through no_grad')
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                x,y=x.type(torch.float32),y.type(torch.float32)
                (x, y) = (x.to("mps"), y.to("mps"))
                # make the predictions and calculate the validation loss
                
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        #avgTrainLoss = avgTrainLoss.item()
        #avgTrainLoss.detach().to('cpu').numpy()[0]
        H["train_loss"].append(avgTrainLoss.item())
        H["test_loss"].append(avgTestLoss.item())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("       Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    
    return unet

def test_unet(unet, images, masks):
    #print('TRAINED WITH Dice Loss on binarised images for 4 epochs')

    unet.eval()

    #num_img = 5

    for i in range(10):
        num_img = i
        image = images[num_img]
        image_tosave = np.copy(image)
        image_input = image.detach().cpu().numpy()
        image_input = np.expand_dims(image_input,-1)
        image_input = np.moveaxis(image_input,-1,0)
        image_input = torch.from_numpy(image_input)
        image_input = image_input.type(torch.float32)
        image_input = image_input.to('mps')
        image_input = image_input.float()
        with torch.no_grad():
            predMask = unet(image_input).squeeze()

        plt.subplot(1,3,1)
        #image = image.detach().cpu().numpy()
        image_tosave = np.moveaxis(image_tosave,0,-1)
        plt.imshow(image_tosave)

        plt.subplot(1,3,2)
        predMask = predMask.detach().cpu().numpy()
        #print(np.unique(predMask))
        predMask[predMask < 0.5] = 0
        #print(np.unique(predMask))
        #predMask = np.moveaxis(predMask,0,-1)
        #print(predMask.shape)
        #predMask = np.moveaxis(predMask,0,1)
        plt.imshow(predMask)

        plt.subplot(1,3,3)
        mask = masks[num_img]
        mask = mask.detach().cpu().numpy()
        #print(np.unique(mask))
        mask = np.moveaxis(mask,0,-1)
        plt.imshow(mask)

        plt.show()

if __name__ == '__main__':

    #Checking if PyTorch MPS (for M1) is activated
    print('PyTorch has MPS activated:',torch.backends.mps.is_built())

    images_path = os.getcwd() + '/812_plate/'
    masks_path = os.getcwd() + '/812_plate_masks/'
    images, masks = get_images_masks(images_path, masks_path,num_imgs=1)

    X_train, X_test, y_train, y_test = get_data(images,masks)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    PIN_MEMORY = False

    trainDS = [(X_train[i],y_train[i]) for i in range(len(X_train))]
    testDS = [(X_test[i],y_test[i]) for i in range(len(X_test))]
    trainLoader = DataLoader(trainDS, shuffle=True,
	                    batch_size=5, pin_memory=PIN_MEMORY,
	                    num_workers=2)

    testLoader = DataLoader(testDS, shuffle=True,
                            batch_size=5, pin_memory=PIN_MEMORY,
                            num_workers=2)
    
    #print(X_train[0].shape)

    unet = train_network(trainLoader, testLoader, NUM_EPOCHS=30)

    test_unet(unet, X_test, y_test)

