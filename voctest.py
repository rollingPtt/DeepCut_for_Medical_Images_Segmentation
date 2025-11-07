import torch
import segment
import util

# parameters
mode = 2
epochs = [10, 100, 15]
step = 1
K = 2
cut = 0
alpha = 3
cc = True
bs = False
log_bin = False
pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
res = (224, 224)
stride = 4
facet = 'key'
layer = 11
in_dir = './VOC2007test/VOC2007/JPEGImages'
out_dir = './results/'
save = True
# Check for mistakes in given arguments
if K != 2 and cc:
    print('largest connected component only available for k == 2')
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

segment.GNN_seg(mode, cut, alpha, epochs, K, pretrained_weights, in_dir, out_dir, save, cc, bs, log_bin, res, facet, layer, stride, device)

