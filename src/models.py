import tensorflow as tf
import ops
import numpy as np
from keras.layers import LSTM
from keras import backend as K
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution1D, AtrousConvolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from layers.subpixel import SubPixel1D, SubPixel1D_v2

def audiounet_processing(x_inp, params):
    n_dim = int(x_inp.get_shape()[1])
    x = x_inp
    L = params['layers']
    r = params['r']
    additive_skip_connection = params['additive_skip_connection']
    n_filtersizes = params['n_filtersizes']
    # dim/layer è sempre : n_dim/2 , n_dim/4, n_dim/8, ...
    #Il numero di canali dipende invece da n_filters.
    if n_dim == 4096:
        print('n_dim = 4096')
        n_filters = params['n_filters_spectral_branch']
    elif n_dim == 8192:
        print('n_dim = 8192')
        n_filters = params['n_filters_time_branch']
    else:
        print('I need other n_dim')
        return None
    downsampling_l = []
    
    # downsampling layers
    for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
            x = (Convolution1D(nb_filter=nf, filter_length=fs,
                  activation=None, border_mode='same', input_dim = (n_dim, 1),
                subsample_length=2, kernel_initializer= tf.orthogonal_initializer()))(x) #init=orthogonal_init, 
            # if l > 0: x = BatchNormalization(mode=2)(x)
            x = LeakyReLU(0.2)(x)
            print('D-Block: {}'.format(x.get_shape()))
            downsampling_l.append(x)
    #print(x)

    # bottleneck layer
    with tf.name_scope('bottleneck_conv'):
        x = (Convolution1D(nb_filter=n_filters[-1], filter_length=n_filtersizes[-1], 
              activation=None, border_mode='same',
              subsample_length=2, kernel_initializer= tf.orthogonal_initializer()))(x) #init=orthogonal_init, 
        x = Dropout(p=0.5)(x)
        x = LeakyReLU(0.2)(x)
        print('B-Block: {}'.format(x.get_shape()))
    # upsampling layers
    for l, nf, fs, l_in in list(zip(range(L), n_filters, n_filtersizes, downsampling_l))[::-1]:
        #print(l, nf, fs, l_in)
        with tf.name_scope('upsc_conv%d' % l):
            # (-1, n/2, 2f)
            x = (Convolution1D(nb_filter=2*nf, filter_length=fs, 
                  activation=None, border_mode='same', kernel_initializer= tf.orthogonal_initializer()))(x) #init=orthogonal_init, 
            x = Dropout(p=0.5)(x)
            x = Activation('relu')(x)
            # (-1, n, f)
            x = SubPixel1D(x, r=2) 
            # (-1, n, 2f)
            x = K.concatenate(tensors=[x, l_in], axis=2)
            print('U-Block: {}'.format(x.get_shape()))
            
    # final conv layer
    with tf.name_scope('lastconv'):
        x = Convolution1D(nb_filter=2, filter_length=9, 
            activation=None, border_mode='same', kernel_initializer= tf.orthogonal_initializer())(x) #init=orthogonal_init, 
        x = SubPixel1D(x, r=2) 
        
    if additive_skip_connection == True:
        x = tf.add(x, x_inp)

    return x #x è (?, 8192). // con tf.reshape(x, [-1, n_dim]) aggiungo un canale alla fine -> diventerebbe (?, 8192, 1)

def tfilm_processing(x_inp, params):
    n_dim = int(x_inp.get_shape()[1])
    DRATE = 2
    # load inputs
    x = x_inp
    L = params['layers']
    r = params['r']
    additive_skip_connection = params['additive_skip_connection']
    n_filtersizes = params['n_filtersizes']
    # dim/layer è sempre : n_dim/2 , n_dim/4, n_dim/8, ...
    #Il numero di canali dipende invece da n_filters.
    if n_dim == 4096:
        print('n_dim = 4096')
        n_filters = params['n_filters_spectral_branch']
    elif n_dim == 8192:
        print('n_dim = 8192')
        n_filters = params['n_filters_time_branch']
    else:
        print('I need other n_dim')
        return None
    downsampling_l = []
    
    # downsampling layers
    for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
            x = (AtrousConvolution1D(nb_filter=nf, filter_length=fs, atrous_rate = DRATE,
                      activation=None, border_mode='same',
                      subsample_length=1, kernel_initializer= tf.orthogonal_initializer()))(x) #init=orthogonal_init
            x = (MaxPooling1D(pool_length=2,border_mode='valid'))(x)
            x = LeakyReLU(0.2)(x)
            # create and apply the normalizer
            nb = 128 / (2**l)
            params_before = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) 
            x_norm = _make_normalizer(x, nf, nb)
            params_after = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) 
            x = _apply_normalizer(x, x_norm, nf, nb)
            print('D-Block: {}'.format(x.get_shape()))
            downsampling_l.append(x)

    # bottleneck layer
    with tf.name_scope('bottleneck_conv'):
        x = (AtrousConvolution1D(nb_filter=n_filters[-1], filter_length=n_filtersizes[-1], atrous_rate = DRATE,
                  activation=None, border_mode='same',
                  subsample_length=1, kernel_initializer= tf.orthogonal_initializer()))(x) #init=orthogonal_init
        x = (MaxPooling1D(pool_length=2,border_mode='valid'))(x)
        x = Dropout(p=0.5)(x)
        x = LeakyReLU(0.2)(x)
        # create and apply the normalizer
        nb = 128 / (2**L)
        x_norm = _make_normalizer(x, n_filters[-1], nb)
        x = _apply_normalizer(x, x_norm, n_filters[-1], nb)
        print('B-Block: {}'.format(x.get_shape()))

    # upsampling layers
    for l, nf, fs, l_in in list(zip(range(L), n_filters, n_filtersizes, downsampling_l))[::-1]:
        with tf.name_scope('upsc_conv%d' % l):
            x = (AtrousConvolution1D(nb_filter=2*nf, filter_length=fs, atrous_rate = DRATE,
                                     activation=None, border_mode='same', kernel_initializer= tf.orthogonal_initializer()))(x)#init=orthogonal_init
            x = Dropout(p=0.5)(x)
            x = Activation('relu')(x)
            x = SubPixel1D(x, r=2) 
            # create and apply the normalizer
            x_norm = _make_normalizer(x, nf, nb)
            x = _apply_normalizer(x, x_norm, nf, nb)
            x = K.concatenate(tensors=[x, l_in], axis=2)
            print ('U-Block: {}'.format(x.get_shape()))
            
    with tf.name_scope('lastconv'):
        x = Convolution1D(nb_filter=2, filter_length=9, 
                              activation=None, border_mode='same', kernel_initializer=tf.keras.initializers.normal())(x)  #, init=normal_init
        x = SubPixel1D(x, r=2) 
    
    if additive_skip_connection == True:
        x = tf.add(x, x_inp)
        
    return x #x è (?, 8192). // con tf.reshape(x, [-1, n_dim]) aggiungo un canale alla fine -> diventerebbe (?, 8192, 1)

def get_function_spectral(net, change):
    if(change == True)&(net == 'tfilm'):
        return get_function('unet')
    elif(change == True)&(net == 'unet'): 
        return get_function('tfilm')
    elif(change == False):
        return get_function(net)

def spectral_processing(audio_lr, params):
    r = params['r']
    L = params['layers']
    f = get_function_spectral(params['net'], params['change_net_in_spectral_branch'])
    # Ramo spettrale -> Funzione audiounet_spectral nel notebook net.py
    with tf.variable_scope('spectral'): #reuse=tf.AUTO_REUSE
        X, Xmag, _ = ops.spectral_transform(audio_lr) 
        if Xmag.get_shape()[-2]%2 == 1: 
            X_dc, X_f = ops.spectral_copies(Xmag, rate = r) #spectral replicator
        else:
            X_dc = None
            X_f =  Xmag
        X_f = f(X_f, params)
        if X_dc is not None:
            NET = tf.concat([X_dc, X_f], axis=1)
        else:
            NET = X_f
    return NET
    
def fusion(time_net, spectral_net):
    # Funzione che mi permette di fare la fusione tra il ramo del dominio temporale e il ramo del dominio spettrale. Sta nel file nets.py
    """Fusion layer to combine predictions from time and frequency branches"""
    #small letters for time, CAPs for frequency
    clamp =lambda x: (tf.tanh(x)+1)/2
    x1 = time_net
    X2 = spectral_net
    with tf.variable_scope('fusion'):
        X1, X1mag, X1arg = ops.spectral_transform(x1)
        Gmag, fusion_weights = ops._timefreq_weighted_average(X1mag, X2, clamp) #fusion_op è ciò che calcola M e w (sarebbero Gmag e fusion_weights)
        G = ops.complexmagarg(Gmag, X1arg) #Gmag è il modulo, X1arg è la fase. complexmagarg restituisce il numero complesso.
        g = ops.inv_spectral_transform(G)
    return g
  
def get_function(net):  
    if net == 'tfilm':
        f = tfilm_processing
    elif net == 'unet':
        f = audiounet_processing
    else:
        print('Please insert a valid net component type. The possibilities are: tfilm and unet.')
        return None
    return f
  
def get_model(audio_lr, params):
    f = get_function(params['net'])
    if params['net_name'] == 'tfilmnet':
        net_pred = f(audio_lr, params)
    elif params['net_name'] in ['gionet', 'gionet2', 'tfnet']:
        time_branch = f(audio_lr, params)
        spec_branch = spectral_processing(audio_lr, params)
        net_pred = fusion(time_branch, spec_branch)
    else:
        print('please insert a valid net_name. The possibilities are: tfilmnet, tfnet, gionet.')
    return net_pred    

def _make_normalizer(x_in, n_filters, n_block):
    """applies an lstm layer on top of x_in"""        
    x_shape = tf.shape(x_in)
    n_steps = tf.cast(x_shape[1], tf.float32)/ n_block # will be 32 at training
    x_in_down = (MaxPooling1D(pool_length=int(n_block), border_mode='valid'))(x_in)
    x_shape = tf.shape(x_in_down)
    x_rnn = LSTM(output_dim = n_filters, return_sequences = True)(x_in_down)
    return x_rnn

def _apply_normalizer(x_in, x_norm, n_filters, n_block):
    x_shape = tf.shape(x_in)
    n_steps = tf.cast(x_shape[1], tf.float32)/ n_block # will be 32 at training
    x_in = tf.reshape(x_in, shape=(-1, n_steps, int(n_block), n_filters))
    x_norm = tf.reshape(x_norm, shape=(-1, n_steps, 1, n_filters))
    x_out = x_norm * x_in
    x_out = tf.reshape(x_out, shape=x_shape)
    return x_out   
   
def normal_init(shape, dim_ordering='tf', name=None):
    return normal(shape, scale=0.0000001, name=name, dim_ordering=dim_ordering)
    
def orthogonal_init(shape, dim_ordering='tf', name=None):
    return orthogonal(shape, name=name, dim_ordering=dim_ordering)