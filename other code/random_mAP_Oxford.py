import argparse
import os
import time
import pickle
import pdb

import numpy as np
import math

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms
import torchvision

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

from PIL import Image
from tqdm import tqdm
from torch.autograd.gradcheck import zero_gradients
from torch.nn.parameter import Parameter
import random

parser = argparse.ArgumentParser(description = "Given images it will calculate the mAP score")
parser.add_argument("-in_dir", "--in_dir", help="Directory for generated adversarial queries.", default="./img_output/")
parser.add_argument("-max_iter", "--max_iter", type=int, help="Maximum iterations.", default="1000")
parser.add_argument("-start_iter", "--start_iter", type=int, help="Start iterations from where to save, make this the same as max iterations if you only want the max.", default="1000")
parser.add_argument("-step_iter", "--step_iter", type=int, help="Iteration steps, make this the same as max iterations if you only want the max.", default="1000")
args = parser.parse_args()

state = torch.load('./models/retrievalSfM120k-resnet101-gem-b80fb85.pth')

net_params = {}
net_params['architecture'] = state['meta']['architecture']
net_params['pooling'] = state['meta']['pooling']
net_params['local_whitening'] = state['meta'].get('local_whitening', False)
net_params['regional'] = state['meta'].get('regional', False)
net_params['whitening'] = state['meta'].get('whitening', False)
net_params['mean'] = state['meta']['mean']
net_params['std'] = state['meta']['std']
net_params['pretrained'] = False


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = 2048

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                    .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                lwhiten.load_state_dict(model_zoo.load_url(L_WHITENING[lw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                    .format(os.path.basename(__file__), lw))

    else:
        lwhiten = None
    
    # initialize pooling
    pool = POOLING['gem']()
    
    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                    .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                rwhiten.load_state_dict(model_zoo.load_url(R_WHITENING[rw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                    .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                    .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'local_whitening' : local_whitening, 
        'pooling' : pooling, 
        'regional' : regional, 
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))

    return net

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)
  
class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'
    
class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr

    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative
class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

pool = GeM
POOLING = {
    'gem'  : GeM}
net = init_network(net_params)
net.load_state_dict(state['state_dict'])

model = net.to('cuda')
model.eval()

import pickle

# ox buildings
with open("./src/gnd_oxford5k.pkl", "rb") as fp:   # Unpickling
    cfg = pickle.load(fp)
im_list = [ii + '.jpg' for ii in cfg['imlist']]

import numpy as np

def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------|
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """
    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1
        #print("pos: ")
        #print(pos)
        #print("len(pos): ")
        #print(len(pos))

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs

normalize = transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
loader = transforms.Compose([transforms.ToTensor(), normalize])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image.cuda() 

def image_loader_re(image_name, scale):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    w, h = image.size
    w, h = round(scale*w), round(scale*h)
    image = image.resize((w, h))
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image.cuda() 

def image_loader_cr(image_name, scale):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    w, h = image.size
    w_n, h_n = round(scale*w), round(scale*h)
    left = (w - w_n)/2
    top = (h - h_n)/2
    right = (w + w_n)/2
    bottom = (h + h_n)/2
    image = image.crop((left, top, right, bottom))
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image.cuda() 

with open("./src/ox_noresize_res101_2048.pkl", "rb") as fp:   # Unpickling
    feat_gem = pickle.load(fp)


location = args.in_dir

sumMap = 0

iterations = 100

for i in tqdm(range(iterations)):
    mAP = []
    query_imgs = []
    MRR = []
    map_tt = []

    adress = location

    for item in cfg['qimlist']:
        im_name = item + '.jpg'

        image = image_loader(adress + im_name)

        queryVec = model(image).data.cpu().numpy()

        feat = np.asarray(feat_gem)

        scores = np.dot(queryVec.T, feat.T[0])
        rank_ID = np.argsort(-1 * scores[0], axis=0)
    #         rank_ID = rank_ID[0][::-1]
        #print(rank_ID)
        #print(type(rank_ID))
        #override the rankid and randomly shuffeling the ranks
        rank_ID = np.random.permutation(rank_ID)
        #print(rank_ID)
        #print(type(rank_ID))

        Q1 = [im_list[int(i)].split('.')[0] for i in rank_ID]

        Q_name = item.split('.')[0][:-6]

        ranked_list = Q1

        query_imgs.append(im_name)
    #         MRR.append(1/ rr)
        map_tt.append(rank_ID)

    #print(np.mean(mAP), np.mean(MRR))
    r1, r2, _, _, = compute_map(np.asarray(map_tt).T, cfg['gnd'])



    print("Iteration " + str(i+1) +" has random mAP: " + str(r1))
    sumMap+= r1
    print("Average random mAP after " + str(i+1) + " iterations: " + str(sumMap/(i+1))
