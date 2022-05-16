import torch
import random
import numpy as np
import logging
import os

#############################
# DataSet Path
#############################
# target_signals_dir = './DataSet/mancini_piano/piano'
target_signals_dir = 'E:\\연구과제\\외래동물\\Segmented_data'
pick_species = ['북방산개구리']
#############################
# Model Params
#############################
model_prefix = 'exp1'
n_iterations = 100000
use_batchnorm = False
lr_g = 1e-4
#lr_d = 3e-4 # you can use with discriminator having a larger learning rate than generator instead of using n_critic updates ttur https://arxiv.org/abs/1706.08500
lr_d = 1e-4
beta1 = 0.5
beta2 = 0.9
decay_lr = False # used to linearly deay learning rate untill reaching 0 at iteration 100,000
generator_batch_size_factor = 1 # in some cases we might try to update the generator with double batch size used in the discriminator https://arxiv.org/abs/1706.08500
# n_critic = 1 # update generator every n_critic steps if lr_g = lr_d the n_critic's default value is 5 
n_critic = 5

# gradient penalty regularization factor.
validate=False
p_coeff = 10
#batch_size = 10
batch_size = 64
noise_latent_dim = 100  # size of the sampling noise
# model_capacity_size = 32    # model capacity during training can be reduced to 32 for larger window length of 2 seconds and 4 seconds
model_capacity_size = 64
# rate of storing validation and costs params
store_cost_every = 1 # 300
progress_bar_step_iter_size = 1 # 400
#############################
# Backup Params
#############################
take_backup = True
backup_every_n_iters = 1000
save_samples_every = 1000
output_dir = 'output'
if not(os.path.isdir(output_dir)):
    os.makedirs(output_dir)
#############################
# Audio Reading Params
#############################
window_length = 16384 #[16384, 32768, 65536] in case of a longer window change model_capacity_size to 32
sampling_rate = 22050 # 16000으로 하게 되면 resample 과정을 거쳐서 속도가 매우 느려짐.
normalize_audio = True 
num_channels = 1

#############################
# Logger init
#############################
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)
#############################
# Torch Init and seed setting
#############################
torch.cuda.set_device(0)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# update the seed
manual_seed = 2019
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
if cuda:
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.empty_cache()