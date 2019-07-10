import argparse
import os
from pathlib import Path
import numpy as np
import random
from PIL import Image
import scipy.misc

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from torch.nn.parameter import Parameter

#progressbar
from tqdm import tqdm

#color difference calculation functions
from color_var import calculateColorDifference

def image_loader(image_name, mean, std, dev, n, sigma):
    """load image, returns tensor, color difference tensor, original mean and original std"""
    print("Loading image: " + str(image_name))
    image = Image.open(image_name)

    #check if the colorThresholds of the image where already calcuated
    filePath = "./pixelVarianceImages/"+ image_name.split("/")[-1].split(".")[0] + ".png"
    savedFile = Path(filePath)

    #define the colorTresh
    colorTresh = None

    if savedFile.is_file():
        #file exists
        print("Found saved color threshold file")
        colorTresh = scipy.misc.imread(filePath)

        #set values back between 0 and 1
        colorTresh = colorTresh/255
    else:
        #file does not exist calculate it
        print("No saved file found, calculating color threshold")
        colorTresh = calculateColorDifference(image, n, sigma)

        #save it as png for filezise reasons
        print("Saving calculated color threshold to " + filePath)
        scipy.misc.imsave(filePath, colorTresh)   

    normalize = transforms.Normalize(mean=mean, std=std)
    loader = transforms.Compose([transforms.ToTensor(), normalize])

    image = loader(image).float()
    colorTresh = torch.from_numpy(colorTresh).float()
    
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(dev), colorTresh.to(dev) 

def img_denorm(img, mean, std, dev, clip=False):
    #convert mean and std from list to np array to do arithmatic
    mean = np.asarray(mean)
    std = np.asarray(std)

    #setup denormalize function
    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0).cpu()
    res = denormalize(res)

    # Attention: here we clipped the resulting image
    if clip:
        res = torch.clamp(res, 0, 1)

    return(res)

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    if p ==np.inf:
        #this is where you change the boundaries based on the pixel
        #torch.max(torch.min(a,b),-b) bounds the v into the threshold,
        # b is threshold vector. The function looks wrong bit it is tested and works
            v = torch.max(torch.min(v,xi), -xi)
    else:
        v = v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v



def data_input_init_sz(xi, h, w, mean, std, dev):
    tf = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])
    
    #initial pertubation vector gets defined here
    #torch.rand returns random numbers between 0 and 1. The 3 is for rgb
    #h and w are trivial. Thus three matrices of with dimentions h and w are made
    #dev indicates the device (CPU,GPU) next balance it between -.5 and .5 
    #finally multiplicate it twice
    v = (torch.rand(1,3,h,w).to(dev)-0.5)*2*xi
    return (tf,v)

def enlarge_to_pixel(new_v, times):
    res = (torch.ceil(torch.abs(new_v) /  0.00390625)  * (torch.sign(new_v))) * 0.004 * times
    return res

#takes the final torch.vector and saves it to an image
def save_to_image(path, image, v, xi, p, percep_optim, mean, std, dev):
    v.data = proj_lp(v.data, xi, p)

    if percep_optim == True:
        large_v = enlarge_to_pixel(v.data, 8)
        modified = image + large_v        
    else:
        modified = image + (10 * v.data)
    
    #denormalize the image
    denormMod = img_denorm(modified, mean, std, dev=dev)

    #save image
    torchvision.utils.save_image(denormMod, path, normalize=False)

def pert_each_im(im_name, model, itr, root, save_dir, dev, percep_optim, treshold, kernelsize, sigma, saveIter):
    #normalization based on imagenet, these have to be calculed from the dataset
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]

    orgImage, colorTresh = image_loader(root + im_name, mean, std, dev, kernelsize, sigma)
    
    image = orgImage
    
    h = image.size()[2]
    w = image.size()[3]

    for param in model.parameters():
        param.requires_grad = False

    p=np.inf
    
    #this defines the actual pixel threshold, the number 255 is chosen such that colorTresh is a bigger number
    xi=colorTresh*treshold/255.0

    tf, init_v = data_input_init_sz(xi, h, w, mean, std, dev=dev)

    v = torch.autograd.Variable(init_v.to(dev),requires_grad=True)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    size = model(torch.zeros(1, 3, h, w).to(dev)).size()

    #since we are using adam the learning rate may start quite high
    learning_rate = 5e-2
    optimizer = torch.optim.Adam([v], lr=learning_rate)
    
    gem_out = model(orgImage)
    loss_track = []

    for t in tqdm(range(itr)):    
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(image + v)

        # Compute the loss. 
        loss =  -1 * loss_fn(y_pred, gem_out)

        loss_track.append(loss.item())
        loss =  -1 * torch.sum(y_pred)

        # reset the gradient since we only want to train the input
        optimizer.zero_grad()

        v.data = proj_lp(v.data, xi, p)

        # calculate the cost gradient
        loss.backward(retain_graph=True)

        # update the input perturbation matrix v
        optimizer.step()

        #save the image in the end to make sure you include the last iteration
        if((t+1)%saveIter == 0 and t != 0):
            #save the image
            path = save_dir + "T" + str(t+1) +"/" + im_name
            save_to_image(path, image, v, xi, p, percep_optim, mean, std, dev)
