import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import warnings
warnings.filterwarnings('ignore')
import argparse
import torchvision.transforms as transforms
import torch
from custom_dataset import custom_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="endoscopy1")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='1792', type=int) 
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.00005', type=float)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--num_layers_PG', default=11, type=int)
parser.add_argument('--num_layers_R', default=10, type=int)
parser.add_argument('--num_R', default=4, type=int)
parser.add_argument('--gpu', default='2', type=str)

parser.add_argument('--input_path', type=str, default='/mnt/data2/yj/mmaction2/data4/')
parser.add_argument('--input_path2', type=str, default='/mnt/data2/yj/mmaction2/data2/')
parser.add_argument('--batch', type=int, default=1)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1

print('-'*20, 'start', '-'*20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
print('Using GPU:', args.gpu)
print('Count of using GPUs:', gpu_count)
# num_workers = 4*int(gpu_count)

image_datasets = {
     'train': 
         custom_dataset(args.input_path + 'train',train=True),
     'test': 
         custom_dataset(args.input_path + 'test',test=True),
     'val':
         custom_dataset(args.input_path +'val',val=True)
         }


dataloaders = {
    'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=args.batch,
                                    shuffle=True,
                                    num_workers=4,
                                    drop_last=True,
                                    ),
    'test':
        torch.utils.data.DataLoader(image_datasets['test'],
                                    batch_size=args.batch,
                                    # shuffle=True,
                                    drop_last=True,
                                    num_workers=4,),
    
    'val':
        torch.utils.data.DataLoader(image_datasets['val'],
                                    batch_size=args.batch,
                                    # shuffle=True,
                                    drop_last=True,
                                    num_workers=4,),
}



vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

model_dir = "./models/"+args.dataset+"/split_"+args.split
results_dir = "./results/"+args.dataset+"/split_"+args.split

load_path = './models/endoscopy/split_1/best_epoch.model'
save_plot_dir = './plot_figures/'+'best_data'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

num_classes = 3


trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split, num_stage=2, num_layer=8)
if args.action == "train":
    trainer.train(model_dir, dataloaders['train'], dataloaders['val'],dataloaders['test'],num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

