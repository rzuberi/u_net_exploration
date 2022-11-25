import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
import tifffile
import numpy as np
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def randomCrop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask

def padding(array, xx, yy):
        h = array.shape[0]
        w = array.shape[1]
        a = (xx - h) // 2
        aa = xx - a - h
        b = (yy - w) // 2
        bb = yy - b - w
        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def get_data(img_path,mask_path,num_imgs):
    full_original_imgs = []
    full_original_masks = []
    for i in range(num_imgs):
        print(str(i) + '/' + str(num_imgs))
        file_path = img_path + '\\' + str(i) + '.tif'
        image = tifffile.imread(file_path)
        image = np.squeeze(image)
        full_original_imgs.append(image)

        file_path = mask_path + '\\' + str(i) + '.tif'
        mask = tifffile.imread(file_path)
        mask = np.squeeze(mask)
        full_original_masks.append(mask)

    initial_data = []
    for i in range(len(full_original_imgs)):
        print(str(i) + '/' + str(len(full_original_imgs)))
        img = full_original_imgs[i]
        images = [img[:, :, 0],img[:, :, 1],img[:, :, 2],img[:, :, 3]]   
        initial_data.append((full_original_masks[i],images))

    cropped_imgs = []
    cropped_masks = []
    
    j = 0
    for data in initial_data:
        print(str(j) + '/' + str(len(initial_data)))
        j += 1
        mask = data[0]
        images = data[1]
        for img in images:
            for i in range(500):
                crop_width = random.randint(5,250)
                crop_height = random.randint(5,250)
                crop_img, crop_mask = randomCrop(img, mask, crop_width, crop_height)
                crop_img = torch.from_numpy(crop_img.astype(np.float))
                crop_mask = torch.from_numpy(crop_mask.astype(np.float))
                cropped_imgs.append(padding(crop_img,250,250))
                cropped_masks.append(padding(crop_mask,250,250))

    return cropped_imgs, cropped_masks

def get_train_test_data(img_path, mask_path, num_imgs, test_size=0.3):
    img_data, mask_data = get_data(img_path, mask_path, num_imgs)
    X_train, X_test, y_train, y_test = train_test_split(img_data, mask_data, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

img_path = os.path.dirname(os.path.realpath(__file__)) + '\812_plate'
mask_path = os.path.dirname(os.path.realpath(__file__)) + '\812_plate_masks'
img_data, mask_data = get_data(img_path, mask_path, 1)

print(img_data[0].shape)

print('Num imgs:',len(img_data))
print('Num masks:',len(mask_data))

X_train, X_test, y_train, y_test = get_train_test_data(img_path, mask_path, 2)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def train(model, opt, loss_fn, epochs, data_loader, print_status):

    loss_ls = []
    epoch_ls = []
    epoch_num = 0
    for epoch in range(epochs):
        avg_loss = 0
        model.train() 

        b=0
        for X_batch, Y_batch in data_loader:
            
            print(X_batch)

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
         
            # set parameter gradients to zero
            opt.zero_grad()
            # print(input_size)
            # forward pass
            Y_pred = model(X_batch)
            
            """
            if (epoch % 10 ==0):
              plt.figure(figsize=(5,5))
              plt.imshow(Y_pred[-1,0,:,:].detach().numpy( ))
            """
            # print('Y_pred shape', Y_pred.shape)
            # print('Y_batch shape before', Y_batch.shape)
            Y_batch = Y_batch.unsqueeze(1)
            Y_batch[1] = Y_pred[1]
            loss = loss_fn(Y_pred, Y_batch)  # compute loss
            loss.backward()  # backward-pass to compute gradients
            opt.step()  # update weights

            # Compute average epoch loss
            avg_loss += loss / len(data_loader)
            #print(b)
            b=b+1
            # print(loss)
        
        """
        if print_status:
            print(f"Loss in epoch {epoch} was {avg_loss}")
        """
        loss_ls.append(avg_loss)
        epoch_ls.append(epoch)
        # Delete unnecessary tensors
        Y_batch[5:] = 0
        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_batch.to(device))).detach().cpu()
        # del X_batch
        Y_hat[5:, 0] = 0
        # plt.subplots_adjust(bottom=1, top=2, hspace=0.2)
        
        print('##epoch ' + str(epoch_num) + '##')
        print('epoch_ls:',epoch_ls,'; training loss loss_ls:', loss_ls)
        #plt.plot(epoch_ls, loss_ls, label='traning loss')
        #plt.legend()
        #plt.xlabel('Epoch'), plt.ylabel('Loss')
        #plt.show()
        epoch_num += 1

    for k in range(4):
      plt.subplot(3, 4, k+1)
      Y_batch2 = Variable(Y_batch[k,0,:,:], requires_grad=False)
      plt.imshow(Y_batch2.cpu().numpy(), cmap='Greys')
      # plt.imshow(X_batch[k,0,:,:].cpu().numpy( ))
      # plt.imshow(Y_batch[k].cpu().numpy( ))
      plt.title('Real')
      plt.axis('off')

      plt.subplot(3, 4, k+5)
      plt.imshow(Y_hat[k, 0], cmap='Greys')
      # plt.imshow(Y_hat[k, 0])
      plt.title('Output')
      plt.axis('off')

    plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
    plt.show()

    return model

def focal_loss(y_pred, y_real, gamma=2):
    y_pred = torch.clamp(F.sigmoid(y_pred), 1e-8, 1-1e-8)
    # gamma = 2
    return -torch.mean(((1-y_pred)**gamma)*y_real*torch.log(y_pred) + (1-y_real)*torch.log(1-y_pred))

batch_size = 4
dataloader_train = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

n_classes = 2
modelUnet = UNet().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizerUnet = optim.Adam(modelUnet.parameters(), lr = 0.000001, weight_decay=1e-2)
num_epochs = 2
model_out = train(modelUnet, optimizerUnet, criterion, num_epochs, dataloader_train, print_status=True)