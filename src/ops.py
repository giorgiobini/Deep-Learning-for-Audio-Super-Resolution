import tensorflow as tf
import numpy as np
from scipy import interpolate

def _timefreq_weighted_average(x1, x2, clamp):
    weight_var = clamp(tf.get_variable('fusion_weights_unclamped', x1.get_shape()[1:]))
    y = x1 * (1-weight_var) + x2 * weight_var #mi sa che è invertito il prodotto di w e (1-w) rispetto al paper.
    return y, weight_var
   
def inv_spectral_transform(X):
    y = irfft(X)
    return y

def spectral_transform(x):
    """rfft on x
    Variable naming conventions:
        CAPs are frequency domain, smalls are time domain
    """
    with tf.name_scope('spectral_transform'):
        X = rfft(x)
        Xmag = tf.abs(X)
        Xarg = arg(X)
    return X, Xmag, Xarg

def rfft(x):
    """
    Ok ho appurato che la dimensione in input di x è (batch_size, dim_x, n_channel)
    """
    """Performs FFT along the axis of interest for Real only signals
    Convenience wrapper around tf.spectral.rfft when for time series with
    multiple channels, in the format (Example, samples, channel)"""
    x = tf.transpose(x, [0, 2, 1])
    x = tf.spectral.rfft(x)
    x = tf.transpose(x, [0, 2, 1])
    return x
    
def arg(x):
    x_r = tf.real(x)
    x_i = tf.imag(x)
    return tf.atan2(x_i, x_r)

def irfft(x):
    x = tf.transpose(x, [0, 2, 1])
    x = tf.spectral.irfft(x)
    x = tf.transpose(x, [0, 2, 1])
    return x

def complexmagarg(_mag, _arg):
    r = tf.cos(_arg)*_mag
    i = tf.sin(_arg)*_mag
    return tf.complex(r, i)

#pylint: disable=invalid-name
def spectral_copies(Xmag, rate=1, expand=True):
    """Apply spectral copies on input Xmag for specified rate

    eg, rate=2
    [ 0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0], [1,2,3,4,1,2,3,4]

    expects data to be in [N, f, c] or [N, t, f, c]

    for rank=4 (i.e spectrograms) number of frequency bins will be expanded if
    expand=True(default). This argument is ignored for rank=3

    rate=1  is degenerate and just splits into dc and non-dc components,
    name_scope is not applied so that it's visible in tensorboard that this
    operation isn't doing anything significant."""
    rank = len(Xmag.get_shape())
    l = int((int(Xmag.shape[-2])-1)/rate) # number of non DC component not zero after LPF
    if rank == 3:
        get_dc = lambda x: x[:, 0, tf.newaxis, :] #tf.newaxis to prevent automatic flatten
        get_passband = lambda x: x[:, 1:l+1, :]
    elif rank == 4:
        get_dc = lambda x: x[:, :, 0, tf.newaxis, :]
        if expand:
            get_passband = lambda x: x[:, :, 1:, :]
        else:
            get_passband = lambda x: x[:, :, 1:l+1, :]

    with tf.name_scope('spectral_copies' if rate > 1 else None):
        X_dc = get_dc(Xmag)
        pass_band = get_passband(Xmag)
        X_f = tf.concat([pass_band
                         for _ in range(rate)],
                        axis=-2)
    return X_dc, X_f

def silence_filtering(sig, top_db):
    #deve restituire solo il segnale filtrato
    filt_sig, _ = librosa.effects.trim(sig, top_db=trim_silence,  frame_length=2048, hop_length=512)
    return filt_sig

def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r #lunghezza del file di output. Corrisponde a len(x), perchè len(x_lr) = len(x)/scale 
    x_sp = np.zeros(x_hr_len)
    i_lr = np.arange(x_hr_len, step=r)  #va da zero a x_hr_len, e fa salti di ampiezza pari a r (scale).
                                        #Es. np.arange(10, step = 3) -> array([0, 3, 6, 9])
    i_hr = np.arange(x_hr_len)
    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)
    return x_sp.astype('float32')