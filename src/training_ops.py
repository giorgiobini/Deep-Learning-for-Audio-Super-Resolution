import tensorflow as tf
import pickle
import re
import os
import numpy as np
import re
import ops
import h5py
from dataset import DataSet
import sys
import pandas as pd
from datetime import datetime
import pytz 
import shutil

def n_epochs_in_multispeaker_task(logdir):
    with open(os.path.join(logdir.replace('speaker1', 'multispeaker'), 'train_metrics.pickle'), 'rb') as handle:
          train_metrics = pickle.load(handle)
    return np.max(list(train_metrics.keys()))

def early_stopping(train_metrics, epochs_for_interruption, multispeaker = True):
    #if validation doesn't improve snr or loss in last epochs_for_interruption then interrupt training
    interrupt = False
    df = pd.DataFrame.from_dict(train_metrics, orient = 'index')
    last_epoch = df.index[-1]
    if df.loc[last_epoch].isnull().values.any() == True: #if there is a NaN in the last epoch metrics,interrupt
        interrupt = True
        print('------- interrupt because there is a NaN in the last epoch metrics -------')
    if multispeaker:
        epoch_of_best_snr_value = np.argmax(df['snr_validation'])
        epoch_of_best_loss_value = np.argmin(df['l2_validation'])
        if (last_epoch - epoch_of_best_snr_value > epochs_for_interruption)&(last_epoch - epoch_of_best_loss_value > epochs_for_interruption):
            interrupt = True
            print('------- interrupt because of the epochs_for_interruption criterion -------')
    return interrupt

def get_file_dicts(data_path):
    file = open(data_path + 'file_list_training', 'rb')
    file_list_tr = pickle.load(file)
    file.close()
    file = open(data_path + 'file_list_validation', 'rb')
    file_list_val = pickle.load(file)
    file.close()
    file = open(data_path + 'file_list_test', 'rb')
    file_list_test = pickle.load(file)
    file.close()
    return file_list_tr, file_list_val, file_list_test

def calc_epochs_from_n_steps(n_steps, batch_size, n_train):
    n_steps_per_epoch = n_train/batch_size
    n_epochs = int(n_steps/n_steps_per_epoch)
    return max(1, n_epochs)

def read_latest_checkpoint(logdir, gdrive):
    ckpt_file = os.path.join(logdir, 'checkpoint')
    read_file = open(ckpt_file, 'rt')
    lines = [line for line in read_file]
    read_file.close()
    if gdrive:
        try:
            old = re.search('(.*)(C:.*Tesi)(.*)', lines[0])[2]
            new = re.search('.*Tesi', logdir)[0]
            lines = [line.replace(old, new) for line in lines]
        except:
            pass
        lines = [line.replace(r'\\', '/') for line in lines]
    else:
        try:
            old = re.search('(.*)(/content.*Tesi)(.*)', lines[0])[2]
            new = re.search('.*Tesi', logdir)[0]
            lines = [line.replace(old, new) for line in lines]
        except:
            pass
        lines = [line.replace('\\', '/') for line in lines]
        lines = [r'{}'.format(line) for line in lines]
    ckpt_out = open(ckpt_file, "w+", encoding = 'utf-8')
    for line in lines:
        ckpt_out.write(line)
    ckpt_out.close()
    checkpoint = tf.train.latest_checkpoint(logdir)
    return checkpoint

def get_logdir(data_path, opt_params, architecture_params):
    log_prefix = os.path.join(data_path, 'model_ckpt')
    net_name = architecture_params['net_name']
    if opt_params['lr_decay']:
        decay = 'dec'
    else:
        decay = ''
    lr_str = '.lr%f' % opt_params['lr'] + decay
    g_str  = '.g%d' % architecture_params['layers']
    b_str  = '.b%d' % int(opt_params['batch_size'])
    r = architecture_params['r']
    ext = net_name + lr_str + '.%d' % r + g_str + b_str
    logdir = os.path.join(log_prefix, ext)
    return logdir

def read_datapath(multispeaker, gdrive):
    if multispeaker:
        dir = 'multispeaker'
    else:
        dir = 'speaker1'
    if gdrive:
        data_path = "/content/gdrive/My Drive/Tesi/processedData/{}/train&validation/".format(dir)
    else:
        ROOT_DIR = os.path.dirname(os.path.abspath('.'))
        data_path = ROOT_DIR + "\\processedData\\{}\\train&validation\\".format(dir)
    return data_path

def read_data(data_path, sample_training = True):
    X_train_d, Y_train = read_hdf5(data_path)
    X_val_d, Y_val = read_hdf5(data_path, train = False)
    if sample_training == True:
        with open(os.path.join(data_path, 'sampling_index.npy'), 'rb') as f:
            sample_index = np.load(f)
        return X_train_d[sample_index,:,:], Y_train[sample_index,:,:], X_val_d, Y_val
    else:
        return X_train_d, Y_train, X_val_d, Y_val
        
def read_hdf5(data_path, train = True):
    if train == True:
        hf = h5py.File(data_path + 'train_data.hdf5', 'r')
    else: #validation
        hf = h5py.File(data_path + 'validation_data.hdf5', 'r')
    return np.array(hf.get('data_lr')), np.array(hf.get('label'))

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
            shape = variable.get_shape()
            var_params = 1
            for dim in shape:
                    var_params *= dim.value
            total_parameters += var_params
    return total_parameters

def create_objective(Y, predictions):
    # load model output and true output
    P = predictions

    # compute l2 loss
    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((P-Y)**2 + 1e-6, axis=[1,2]))
    sqrn_l2_norm = tf.sqrt(tf.reduce_mean(Y**2, axis=[1,2]))
    snr = 20 * tf.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.log(10.)

    avg_sqrt_l2_loss = tf.reduce_mean(sqrt_l2_loss, axis=0)
    avg_snr = tf.reduce_mean(snr, axis=0)

    # save losses into collection
    tf.add_to_collection('l2_on_batch', avg_sqrt_l2_loss) #potrei anche non salvarla, tanto la restituisco nel train 
    tf.add_to_collection('snr_on_batch', avg_snr)

    return avg_sqrt_l2_loss
    
def get_params():
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                     if 'soundnet' not in v.name ]

def create_optimzier(opt_params):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    tf.add_to_collection('global_step', global_step)
    
    if opt_params['alg'] == 'adam':
        b1, b2 = opt_params['b1'], opt_params['b2']
        if opt_params['lr_decay']:
            lr = tf.train.polynomial_decay(opt_params['lr'],
                                                            end_learning_rate=1e-6,
                                                            global_step=global_step,
                                                            decay_steps=500000,
                                                            power=0.5)
        else:
            lr = opt_params['lr']

        optimizer = tf.train.AdamOptimizer(lr, b1, b2, epsilon=1e-6, name='MyAdam')
    else:
        raise ValueError('Invalid optimizer: ' + opt_params['alg'])
        
    return optimizer

def create_gradients(loss, params, optimizer):
    gv = optimizer.compute_gradients(loss, params)
    g, v = zip(*gv)
    return g

def create_updates(params, grads, alpha, optimizer):
    
    # create a variable to track the global step.
    global_step = tf.get_collection('global_step')[0]

    # update grads
    grads = [alpha*g for g in grads]

    # use the optimizer to apply the gradients that minimize the loss
    gv = zip(grads, params)
    train_op = optimizer.apply_gradients(gv, global_step = global_step)

    return train_op

def load_batch(inputs, batch, alpha=1, train=True):
    X_in, Y_in, alpha_in = inputs
    X, Y = batch
    if Y is not None:
        feed_dict = {X_in : X, Y_in : Y, alpha_in : alpha}
    else:
        feed_dict = {X_in : X, alpha_in : alpha}
    g = tf.get_default_graph()
    k_tensors = [n for n in g.as_graph_def().node if 'keras_learning_phase' in n.name] 
    if k_tensors: 
        k_learning_phase = g.get_tensor_by_name(k_tensors[0].name + ':0')
        feed_dict[k_learning_phase] = train #non ho capito a cosa serve questa chiave k_learning_phase.                
    return feed_dict

def train(feed_dict, train_op, loss, sess):
    _, loss = sess.run([train_op, loss], feed_dict=feed_dict) #_ serve per fare gli aggiornamenti dei pesi, ma ÃƒÆ’Ã‚Â¨ nullo come valore
    return loss

def print_epoch_with_time(tot_time_s, epoch_number):
    it = pytz.timezone('Europe/Paris') 
    current_time = datetime.now(it).strftime('%H:%M')
    if (tot_time_s > 60)&(tot_time_s<36000):
        tot_time_m = tot_time_s/60
        print('------- EPOCH {} RESULTS ------- (trained in {} {}. Current time: {})'.format(epoch_number, 
                                                                         np.round(tot_time_m, 2), 'minutes',
                                                                         current_time))
    elif (tot_time_s>36000):
        tot_time_h = tot_time_s/36000
        print('------- EPOCH {} RESULTS ------- (trained in {} {}. Current time: {})'.format(epoch_number, 
                                                                         np.round(tot_time_h, 2), 'hours',
                                                                         current_time))
    else:
        print('------- EPOCH {} RESULTS ------- (trained in {} {}. Current time: {})'.format(epoch_number, 
                                                                         np.round(tot_time_s, 2), 'seconds',
                                                                         current_time))
                                                                         
def print_metrics(l2_training, snr_training, l2_validation, snr_validation):
    print("l2 (loss) on training is {}".format(l2_training))
    print("snr on training is {}".format(snr_training))
    print("l2 (loss) on validation is {}".format(l2_validation))
    print("snr on validation is {} \n".format(snr_validation))
        
def save_model_phase(sess, logdir, epochs_completed, train_metrics, saver, multispeaker = True):
    checkpoint_root = os.path.join(logdir, 'model_ckpt')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_root, global_step=epochs_completed)
    with open(os.path.join(logdir, 'train_metrics.pickle'), 'wb') as handle:
        pickle.dump(train_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    if multispeaker:
        save_best_model(logdir, train_metrics)
    remove_old_logs(logdir, epochs_completed)

def save_best_model(logdir, train_metrics):
    df = pd.DataFrame.from_dict(train_metrics, orient = 'index')
    epoch_of_best_snr_value = np.argmax(df['snr_validation'])
    epoch_of_best_loss_value = np.argmin(df['l2_validation'])
    epoch_of_best_model = max(epoch_of_best_snr_value, epoch_of_best_snr_value)
    for f in os.listdir(logdir):
        #print(f)
        if re.search('model.ckpt-{}'.format(epoch_of_best_model), f):
            name, ext = f.split('.')
            src=os.path.join(logdir, f)
            new_name = 'best_model.' + ext
            dst=os.path.join(logdir, new_name)
            shutil.copy(src,dst)
    
def remove_old_logs(logdir, current_epoch):
    for f in os.listdir(logdir):
        #print(f)
        if re.search('model.ckpt-{}'.format(current_epoch-5), f): #tengo solo gli ultimi 5 log
            os.remove(os.path.join(logdir, f))    
    
def get_epochs_completed(checkpoint):
    m = re.search('\d*$',checkpoint)
    if len(m.group(0))>0:
        epochs_completed = int(m.group(0))
    else:
        epochs_completed = 0
    return epochs_completed