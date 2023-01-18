import argparse
import numpy as np
import random

import sys 
sys.path.append('./utils/') 

# custom libaries
import utils
from training import training

############### 
######ARGUMENTS
# define parser
parser = argparse.ArgumentParser(description='training of a EPiC-GAN --- equivariant point cloud generative adversarial network')

# add possible argument

## OFTEN CHANGED
parser.add_argument('--equiv_layers_generator', '-el', default=6, help='number of equivariant layers for the generator', type=int)
parser.add_argument('--equiv_layers_discriminator', '-eld', default=3, help='number of equivariant layers for the discriminator', type=int)
parser.add_argument('--latent', '-l', default=10, help='number of global latent variables', type=int)
parser.add_argument('--latent_local', '-ll', default=3, help='number of local latent variables', type=int)
parser.add_argument('--epochs', '-e', default=10, help='number of epochs to train', type=int)
parser.add_argument('--save_folder', '-sf', 
                    default='/beegfs/desy/user/buhmae/2_PointCloudGeneration/220727_GANscan/'
                    , help='folder to save trainings in', type=str)
parser.add_argument('--n_points', '-n', default=150, help='number of particles', type=int)
parser.add_argument('--dataset_type', default='jetnet_top', help='define the dataset type', type=str )


## define dataset type, model name (and namef of save file)
parser.add_argument('--project_prefix', type=str, default='EPiC-GAN_orderGlobalLocal_', help='for project naming on W$B or comet.ml')
parser.add_argument('--add_jet_type_fpnd', default='q', type=str)
parser.add_argument('--reason', default='3 EPiC discr. fixed, 2000eps, loc=3, hid=64', type=str, help='explain reason for running this run')
parser.add_argument('--GAN_type', default='LSGAN', type=str, help='LSGAN or GAN')


# optimizer
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate of generator optimizer')
parser.add_argument('--lr_C', default=1e-4, type=str, help='learning rate of discrminiator (critic) optimizer')
parser.add_argument('--beta1', default=0.9, type=float, help='adam parameter, default: 0.9')
# further model parameters
parser.add_argument('--hid_d', default=128, type=int, help='hidden dimensionality of model layers, default: 128')
parser.add_argument('--batch_size_max', default=128, type=int, help='maximum batch size')
parser.add_argument('--feats', default=3, type=int, help='number of features, for jets =3 (pt,rapidity,phi)')
# preprocessing
parser.add_argument('--normalize_points', default=True, type=bool, help='standardisation of points enabled, default: True')
parser.add_argument('--normalize_points_forDiscrOnly', default=False, type=bool, help='standardisation enabled only for discrmiinator, default: False')
parser.add_argument('--norm_sigma', default=5, type=int, help='standardisation with sigma X (with of normal distibution, default: 5')
parser.add_argument('--center_gen', default=True, type=bool, help='center generation (for evaluation)')
## logging arguments
parser.add_argument('--log_interval', default=250, type=int, help='interval for wandb loggging')
parser.add_argument('--wandb_dir', default='/beegfs/desy/user/buhmae/2_PointCloudGeneration/', type=str, help='wandb folder')
parser.add_argument('--log_wandb', default=False, type=bool, help='enable wandb logging')
parser.add_argument('--log_comet', default=False, type=bool, help='enable comet logging')
parser.add_argument('--save_interval', default=1000, type=int, help='intervall for model weights saving (latest model saved)')

parser.add_argument('--model_name', default='EPiC_GAN', type=str, help='model name')

# read parser into class
params = parser.parse_args()

# assign random number to run
rand = random.randint(11111,99999)  
parser.rand = rand

params.save_file_name  = params.GAN_type+'_'+params.model_name+'_el'+str(params.equiv_layers_generator)+'_l'+str(params.latent)+'_'+str(rand)

print('\n\n\n\n Parameters: ')
print(vars(params), '\n\n\n\n\n')  # convert parser class to dict

############### 
#######TRAINING
best_w_dist_ms, test_w_dist_ms, best_epoch, epoch_time = training(params)

out_list = [rand, params.equiv_layers_generator, params.latent, 
            params.latent_local, params.epochs,
            best_w_dist_ms, test_w_dist_ms, best_epoch, epoch_time]
print(out_list)
      
      
### SAVE OUTPUT in TXT
utils.create_folder_ifNotExists(path='./output/')
if params.dataset_type == 'ef_quark':
    txtfile = './output/'+params.model_name+'_'+str(params.n_points)+'.txt'
else:
    txtfile = './output/'+params.model_name+'_'+params.dataset_type+'_'+str(params.n_points)+'.txt'
header = ('rand equiv_layers_generator latent latent_local epochs best_w_dist_ms test_w_dist_ms best_epoch last_epoch_time')
ary = np.array([out_list])
utils.savetxt(txtfile, ary, header=header)




