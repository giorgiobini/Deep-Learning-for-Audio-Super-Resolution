import tensorflow as tf
from training_ops import load_batch
from scipy.signal import decimate
from ops import upsample
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import pandas as pd
import pickle
import os
import librosa
from scipy import signal
from dataset import *
import deepspeech
import time
import subprocess
import shlex
from shlex import quote
import jiwer

def df_from_results_stt(results, 
                        savepath = None):
    all_results = pd.DataFrame.from_dict(results, 'index')
    means = all_results.mean()
    stds = all_results.std()
    speech_to_text_results = pd.DataFrame({'mean': means, 'std': stds}).T
    if savepath:
        name_path = os.path.join(savepath, 'speech_to_text_results.csv')
        speech_to_text_results.to_csv(name_path, index=True)
        print('DataFrame of speech to text results is successfully saved!')
    return speech_to_text_results

def calc_wer(ground_truth, hypothesis):
    transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation()
    ]) 
    wer = jiwer.wer(
        ground_truth, 
        hypothesis, 
        truth_transform=transformation, 
        hypothesis_transform=transformation
    )
    return wer

def from_txt_to_list(txt):
    out_list = []
    with open(txt) as f:
        for line in f:
            out_list.append(line.strip())
    return out_list

def get_deepspeech_model(path, lm_alpha = 0.931289039105002, lm_beta = 0.931289039105002, beam_width = 1024):
    model_file_path = os.path.join(path, 'deepspeech-0.9.3-models.pbmm')
    scorer_file_path = os.path.join(path,'deepspeech-0.9.3-models.scorer')
    model = deepspeech.Model(model_file_path)
    model.enableExternalScorer(scorer_file_path)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)
    return model
    
def get_txt(audio_file, model, fs = 16000, gdrive = True):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_file), fs)
    if (gdrive == False):
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE, shell = True)
        #proc = subprocess.Popen(shlex.split(sox_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #out, err = proc.communicate(timeout=120)
    else:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE, shell = False)
    data16 = np.frombuffer(output, np.int16)
    text = model.stt(data16.astype('int16'))
    return text

def plot_stft_spectogram(input_sig, rate = 16000, window = 'hamming', nperseg = 512, noverlap = 256, limit_frequency_to_plot = np.inf):
    f, t, Sxx = signal.spectrogram(input_sig, fs = rate, window = window, nperseg = 512, noverlap = 256)
    idx = len(f[f<=limit_frequency_to_plot])
    Sxx_ = 10*np.log10(Sxx[:idx, ]) #Sxx[:idx, ]
    f_ = f[:idx, ]
    plt.colorbar(plt.pcolormesh(t, f_, Sxx_, shading='gouraud')).set_label('Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def df_from_results(train_snr = None, 
                    train_snr_std = None,
                    val_snr = None, 
                    val_snr_std = None,
                    test_snr = None,
                    test_snr_std = None,
                    train_lsd = None, 
                    train_lsd_std = None, 
                    val_lsd = None,
                    val_lsd_std = None,
                    test_lsd = None, 
                    test_lsd_std = None, 
                    savepath = None):
    diz = {'train': {'snr_mean': train_snr,
                    'snr_std': train_snr_std,
                    'lsd_mean':train_lsd,
                    'lsd_std':train_lsd_std
                    },
            'val': {'snr_mean': val_snr,
                    'snr_std': val_snr_std,
                    'lsd_mean':val_lsd,
                    'lsd_std':val_lsd_std
                    },
            'test': {'snr_mean': test_snr,
                    'snr_std': test_snr_std,
                    'lsd_mean':test_lsd,
                    'lsd_std':test_lsd_std
                    }
            }
    df = pd.DataFrame.from_dict(diz, orient = 'index')
    if savepath:
        name_path = os.path.join(savepath, 'df_results.csv')
        df.to_csv(name_path, index=True)
        print('DataFrame is successfully saved!')
    return df
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return np.array(out)

class ModelOnSingleFile:
    def __init__(self, sig, sampling_rate, r, sess, inputs, predictions, interp = True):
        self.rate = sampling_rate
        self.inputs = inputs
        self.predictions = predictions
        self.sess = sess
        patch_dimension = int(inputs[0].shape[1])
        
        
        num_to_keep = int(np.floor(len(sig) / patch_dimension) * patch_dimension)
        sig = sig[:num_to_keep]
        sig = sig[ : len(sig) - (len(sig) % r)]  # Es: scaling_factor = 2 -> se il numero di campioni (lunghezza di x) ÃƒÂ¨ pari, allora non succede nulla. Se ÃƒÂ¨ dispari, invece, l'ultimo campione viene rimosso. 
        sig_lr = decimate(sig, r)
        
        if interp:
            sig_lr = upsample(sig_lr, r)
            assert len(sig_lr) == len(sig)
        
        num_y = int(sig.shape[0]/patch_dimension)       
        #generate patches
        self.Y = np.expand_dims(chunkIt(sig, num_y), axis = -1)
        if interp:
            self.X_lr = np.expand_dims(chunkIt(sig_lr, num_y), axis = -1)
        else:
            self.X_lr = np.expand_dims(chunkIt(sig_lr, int(num_y/r)), axis = -1)
        
        self.batches = (self.X_lr, self.Y)
    
    def get_model_hr(self):
        feed_dict_v = load_batch(self.inputs, self.batches, train = False)
        result = np.squeeze(self.sess.run(self.predictions, feed_dict = feed_dict_v)).flatten()
        return result
    
    def low_res(self):
        return self.X_lr.flatten()
    
    def original_version(self):
        return self.Y.flatten()

def plot_training_curves(train_curve, validation_curve, y_axis = 'loss', train_color = '#2D4059', validation_color = '#FFB400'):
    sns.set_style("darkgrid", {"axes.facecolor": "#F6F6F6"})
    sns.set_context("notebook", font_scale=1.25)
    epoch = np.arange(len(train_curve)) + 1
    plt.plot(epoch, train_curve, label = 'train', color = train_color)
    plt.plot(epoch, validation_curve, label = 'validation', color = validation_color)
    plt.title('Model {}'.format(y_axis))
    plt.xlabel('epoch')
    plt.ylabel(y_axis)
    #plt.grid(True, which='both')
    plt.legend()
    plt.show()
    
def calc_snr(Y, Pred):
    sqrt_l2_loss = np.sqrt(np.mean((Pred-Y)**2+1e-6, axis=(0, 1)))
    sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(0,1)))
    snr = 20* np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
    return snr

def calc_snr2(Y, P, return_std = False):
    sqrt_l2_loss = np.sqrt(np.mean((P-Y)**2 + 1e-6, axis=(1,2)))
    sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(1,2)))
    snr = 20 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
    avg_snr = np.mean(snr, axis=0)
    std_snr = np.std(snr, axis=0)
    if return_std:
        return avg_snr, std_snr
    else:
        return avg_snr

def calc_snr_lsd(input_x, input_y, inputs, predictions, sess, alpha = 1, batch_size=128):
    assert input_x.shape[0]%batch_size == 0
    data = DataSet(input_x, input_y, epochs_completed = 0)
    snr_on_batch = []
    snr_on_batch_std = []
    lsd_on_batch = []
    lsd_on_batch_std = []
    for i in tqdm(range(int(input_x.shape[0]/batch_size))): # 1 epoch
        batch = data.next_batch(batch_size)
        Y_batch = batch[1]
        feed_dict_not_train = load_batch(inputs, batch, alpha, train=False)
        result_tr = sess.run(predictions, feed_dict = feed_dict_not_train)
        snr_mean, snr_std = calc_snr2(Y_batch, result_tr, return_std = True)
        snr_on_batch.append(snr_mean)
        snr_on_batch_std.append(snr_std)
        lsd_list = [compute_log_distortion(np.squeeze(result_tr[i]), np.squeeze(Y_batch[i])) for i in range(result_tr.shape[0])]   
        lsd_on_batch.append(np.mean(lsd_list))
        lsd_on_batch_std.append(np.std(lsd_list))
    return np.mean(snr_on_batch), np.mean(snr_on_batch_std), np.mean(lsd_on_batch), np.mean(lsd_on_batch_std) 
    
def get_power(x):
    S = librosa.stft(x, 2048)
    p = np.angle(S)
    S = np.log(np.abs(S)**2 + 1e-8)
    return S

def calc_lsd(y, p):
    Y, P = np.squeeze(y), np.squeeze(p)
    return compute_log_distortion(Y, P)

def compute_log_distortion(x_hr, x_pr):
        S1 = get_power(x_hr)
        S2 = get_power(x_pr)
        lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis = 0)
        return min(lsd, 10.)
    
def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)

def pre_process_signal(sig, fs = 16000):
    l = sig.shape[0]
    div = np.floor(l/8192)
    diff = int(8192*(div+1) - l)
    noise = band_limited_noise(min_freq=300, max_freq = 5000, samples=diff, samplerate=fs)
    assert len(noise) == diff
    sig = np.concatenate([sig, noise], axis = 0)
    return sig