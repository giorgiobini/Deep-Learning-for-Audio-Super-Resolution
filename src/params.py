import pickle
import os
gionet_architecture_params = {'net_name': 'gionet',
                       'r': 4,
                       'layers': 3,
                       'n_filtersizes': [65, 33, 17,  9,  9,  9,  9, 9, 9],
                       'n_filters_time_branch': [64, 128, 256, 256],#[64, 128, 192, 256],
                       'n_filters_spectral_branch':[64, 128, 256, 256], #[64, 128, 192, 256], 
                       'net': 'tfilm', #choose between 'tfilm' and 'unet'
                       'change_net_in_spectral_branch': False, #if true and 'net': 'tfilm'/'unet', then time_net is 'tfilm' and spectral_branch is 'unet'/'tfilm'.
                       'additive_skip_connection': True
                      }
gionet_opt_params = {'alg': 'adam', 'lr': 3e-5, 'lr_decay': True, 'b1': 0.99, 'b2': 0.999, 'batch_size': 128, 'tot_steps': 500000, 'steps_for_interruption':25000}

gionet2_architecture_params = {'net_name': 'gionet2',
                       'r': 4,
                       'layers': 3,
                       'n_filtersizes': [65, 33, 17,  9,  9,  9,  9, 9, 9],
                       'n_filters_time_branch': [64, 128, 256, 256],
                       'n_filters_spectral_branch': [64, 128, 288, 288],
                       'net': 'tfilm', #choose between 'tfilm' and 'unet'
                       'change_net_in_spectral_branch': True, #if true and 'net': 'tfilm'/'unet', then time_net is 'tfilm' and spectral_branch is 'unet'/'tfilm'.
                       'additive_skip_connection': True
                      }
gionet2_opt_params = {'alg': 'adam', 'lr': 3e-5, 'lr_decay': True, 'b1': 0.99, 'b2': 0.999, 'batch_size': 128, 'tot_steps': 500000, 'steps_for_interruption':25000}

tfilmnet_architecture_params = {'net_name': 'tfilmnet',
                       'r': 4,
                       'layers': 3,
                       'n_filtersizes': [65, 33, 17,  9,  9,  9,  9, 9, 9],
                       'n_filters_time_branch': [64, 128, 448, 448],#[64, 128, 512, 512]
                       'net': 'tfilm', #choose between 'tfilm' and 'unet'
                       'additive_skip_connection': True
                      }
tfilmnet_opt_params = {'alg': 'adam', 'lr': 1e-5, 'lr_decay':False, 'b1': 0.99, 'b2': 0.999, 'batch_size': 128, 'tot_steps': 500000, 'steps_for_interruption':25000}

tfnet_architecture_params = {'net_name': 'tfnet',
                       'r': 4,
                       'layers': 3,
                       'n_filtersizes': [65, 33, 17,  9,  9,  9,  9, 9, 9],
                       'n_filters_time_branch': [64, 128, 288, 288],
                       'n_filters_spectral_branch': [64, 128, 288, 288],
                       'net': 'unet', #choose between 'tfilm' and 'unet'
                       'change_net_in_spectral_branch': False, #if true and 'net': 'tfilm'/'unet', then time_net is 'tfilm' and spectral_branch is 'unet'/'tfilm'.
                       'additive_skip_connection': True
                      }
tfnet_opt_params = {'alg': 'adam', 'lr': 3e-5, 'lr_decay':True, 'b1': 0.99, 'b2': 0.999, 'batch_size': 128, 'tot_steps': 500000, 'steps_for_interruption':25000}

def save_architecture_opt_params(log_dir, architecture_params, opt_params):
    with open(os.path.join(log_dir, 'params.pickle'), 'wb') as handle:
        pickle.dump((architecture_params, opt_params), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def read_params_from_logdir(log_dir):
    with open(os.path.join(log_dir, 'params.pickle'), 'rb') as handle:
        architecture_params, opt_params = pickle.load(handle)
    return architecture_params, opt_params
    
def read_params(name):
    assert name in ['gionet', 'gionet2', 'tfnet', 'tfilmnet']
    if name == 'gionet':
        return gionet_architecture_params, gionet_opt_params
    if name == 'gionet2':
        return gionet2_architecture_params, gionet2_opt_params
    if name == 'tfilmnet':
        return tfilmnet_architecture_params, tfilmnet_opt_params
    if name == 'tfnet':
        return tfnet_architecture_params, tfnet_opt_params
    