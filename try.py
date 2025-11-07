import os
import torch
os.environ['TORCH'] = torch.__version__
# print(torch.__version__)

import segment
import util

mode = 2
epochs = [10, 100, 15]
step = 1
K = 6
cut = 0
alpha = 2.5
# cc can be True only when k == 2 (It means that we show only the largest component in segmentation map)
cc = False
# apply bilateral solver
bs = False
# Apply log binning to extracted descriptors (correspond to smoother segmentation maps)
log_bin = False
# Directory to pretrained Dino
pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
# Resolution for dino input, higher res != better performance as Dino was trained on (224,224) size images
res = (224, 224)
# stride for descriptor extraction
stride = 4
# facet fo descriptor extraction (key/query/value)
facet = 'key'
# layer to extract descriptors from
layer = 11
# Directory of image to segment
in_dir = './images/Synapsedata/'
out_dir = './results/'
save = True
# Check for mistakes in given arguments
if K != 2 and cc:
    print('largest connected component only available for k == 2')
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# Single Object Segmentation
segment.GNN_seg(mode, cut, alpha, epochs, K, pretrained_weights, in_dir, out_dir, save, cc, bs, log_bin, res, facet, layer, stride, device)