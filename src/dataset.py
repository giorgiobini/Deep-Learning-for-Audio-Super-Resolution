"""Class for doing iterations over datasets

This is stolen from the tensorflow tutorial
"""

import numpy as np
import os
import h5py
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# ----------------------------------------------------------------------------


class DataSetGenerator(object):
  def __init__(self,
               path,
               file_diz,
               epochs_completed = 0,
               dtype=dtypes.float32):
    """
    Construct a DataSet from data in a folder (path).
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    
    self._path = path
    self._file_diz = file_diz
    self._file_list = np.array(np.array(list(file_diz.keys())))
    self._num_examples = len(file_diz.keys())
    self._epochs_completed = epochs_completed
    self._index_in_epoch = 0
    
  @property
  def path(self):
    return self._path
    
  @property
  def file_list(self):
    return self._file_list
    
  @property
  def file_diz(self):
    return self._file_diz

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
    
  def load_full_batch(self, shuffle = True):
    return load_batch_data(self.path, self.file_list, self._file_diz)
   
  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._file_list = self.file_list[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      file_list_rest_part = self._file_list[start:self._num_examples]
      datapoints_rest_part, labels_rest_part = load_batch_data(self._path, file_list_rest_part, self._file_diz)
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._file_list = self.file_list[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      file_list_new_part = self._file_list[start:end]
      datapoints_new_part, labels_new_part = load_batch_data(self._path, file_list_new_part, self._file_diz)
      return np.concatenate((datapoints_rest_part, datapoints_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return load_batch_data(self._path, self._file_list[start:end], self._file_diz)

def load_batch_data(path, file_names, diz):
    if len(file_names)>0:
        datapoints = []
        labels = []
        for file_name in file_names:
            fold1 = diz[file_name]['fold1']
            nested_fold = str(diz[file_name]['nested_fold'])
            hf = h5py.File(os.path.join(path, fold1, nested_fold, file_name), 'r')
            datapoints.append(np.array(hf.get('data_lr')))
            labels.append(np.array(hf.get('label')))
        datapoints, labels = np.array(datapoints), np.array(labels)
        assert datapoints.shape == labels.shape
        return np.expand_dims(datapoints, axis = -1), np.expand_dims(labels, axis = -1)
    else:
        return np.empty([0,8192,1]), np.empty([0,8192,1])


class DataSet(object):

  def __init__(self,
               datapoints,
               labels,
               epochs_completed = 0,
               tl_input = None,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    
    G: tl_input is the transfer learning input.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    if labels is None:
      labels = np.zeros((len(datapoints),))

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert datapoints.shape[0] == labels.shape[0], (
          'datapoints.shape: %s labels.shape: %s' % (datapoints.shape, labels.shape))
      self._num_examples = datapoints.shape[0]

    self._datapoints = datapoints
    self._labels = labels
    self._epochs_completed = epochs_completed
    self._index_in_epoch = 0
    self._tl_input = tl_input
    self._is_multi_input = False
    if type(self._tl_input) is np.ndarray:
        self._is_multi_input = True

  @property
  def datapoints(self):
    return self._datapoints

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def tl_input(self):
    return self._tl_input

  @property
  def ismultinput(self):
    return self._is_multi_input

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._datapoints = self.datapoints[perm0]
      self._labels = self.labels[perm0]
      if self._is_multi_input:
          self._tl_input = self._tl_input[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      datapoints_rest_part = self._datapoints[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      if self._is_multi_input:
          tl_input_rest_part = self._tl_input[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._datapoints = self.datapoints[perm]
        self._labels = self.labels[perm]
        if self._is_multi_input:
            self._tl_input = self.tl_input[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      datapoints_new_part = self._datapoints[start:end]
      labels_new_part = self._labels[start:end]
      if self._is_multi_input:
          tl_input_new_part = self._tl_input[start:end]
          return (np.concatenate((datapoints_rest_part, datapoints_new_part), axis=0), np.concatenate((tl_input_rest_part, tl_input_new_part), axis=0)), np.concatenate((labels_rest_part, labels_new_part), axis=0)
      else:
          return np.concatenate((datapoints_rest_part, datapoints_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      if self._is_multi_input:
          return (self._datapoints[start:end], self._tl_input[start:end]), self._labels[start:end]
      else:
          return self._datapoints[start:end], self._labels[start:end]
