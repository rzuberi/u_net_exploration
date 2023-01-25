#lets get cellpose and try to access some layers

from math import log as ln
import torch
import cellpose
from cellpose import models
from cellpose import core
from cellpose.io import imread
import os
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
import numpy as np

#Check if GPU is activated
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[core.use_gpu()]} \n')

files = [os.getcwd() + '\\812_plate\\' + str(i) + '.tif' for i in range(2)]
imgs = [imread(f) for f in files]

#print(imgs[0].shape)

model = models.CellposeModel(model_type='nuclei',gpu=core.use_gpu())
#print(model)
#print(model.gpu)
masks, flows, styles = model.eval(imgs,channels=[[0,0]],cellprob_threshold=False)
#print(masks[0].shape)
#print(masks[0])
#plt.imshow(masks[0])
#plt.show()
#print('\n')

#print(type(flows[0][2]))
#plt.imshow(flows[0][2] > 0.0)
#plt.show()
print('\n')
print(flows[0][2])
print('\n')
print(flows[1][2])
#print(np.unique(flows[0][2]))

#print('n_epochs',model.n_epochs)

#x = -10.557406
#rev = ln(x/(1-x))
#print(rev)