# Adapted from https://github.com/hojonathanho/diffusion to work on multiple inputs
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class IXI_dataset:
  def __init__(self,
    tfr_file_us_image,            # Path to tfrecord file.
    tfr_file_mask,
    label_file,
    resolution=256,      # Dataset resolution.
    max_images=7500,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
    batch_size=1,
  ):
    """Adapted from https://github.com/NVlabs/stylegan2/blob/master/training/dataset.py.
    Use StyleGAN2 dataset_tool.py to generate tf record files.
    """
    self.tfr_file_us_image  = tfr_file_us_image
    self.tfr_file_mask      = tfr_file_mask
    self.label_file         = label_file
    self.dtype              = 'float32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]
    self.batch_size = batch_size

  def train_input_fn(self):
    # Build TF expressions.
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    # old line
    # dset_us = dset_us.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # new line
    dset_us = dset_us.map(lambda r: self._parse_tfrecord_tf(r, 'us_im'), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_us = dset_us.take(self.max_images)

    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    # old line
    # dset_mask = dset_mask.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # new line
    dset_mask = dset_mask.map(lambda r: self._parse_tfrecord_tf(r, 'mask'), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_mask = dset_mask.take(self.max_images)

    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)    
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})
    _tf_labels_dataset = _tf_labels_dataset.take(self.max_images)

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask))
    # old line
    # dset = dset.map(lambda x,y: dict(us_im= x["us_im"], mask=y["mask"]))
    # new line
    dset = dset.map(lambda x, y: {"us_im": x["us_im"], "mask": y["mask"]})

    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))
    # Repeat dataset infinitely so training can continue until max_steps
    dset = dset.repeat()
    # Shuffle and prefetch
   # dset = dset.shuffle(50000)
    dset = dset.batch(self.batch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset



  def eval_input_fn(self):
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    dset_us = dset_us.map(lambda r: self._parse_tfrecord_tf(r, 'us_im'), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    dset_mask = dset_mask.map(lambda r: self._parse_tfrecord_tf(r, 'mask'), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask))
    dset = dset.map(lambda x,y: dict(us_im=x["us_im"], mask=y["mask"]))
    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))

    dset = dset.batch(self.batch_size, drop_remainder=True)
    # new code - mapping correctly the channels
    def force_shapes(features, labels):
        # Force the us_im to [16, 2, 256, 256]
        features['us_im'] = tf.reshape(features['us_im'], [self.batch_size, 2, 256, 256])
        # Force the mask to [16, 1, 256, 256] 
        features['mask'] = tf.reshape(features['mask'], [self.batch_size, 1, 256, 256])
        return features, labels

    dset = dset.map(force_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # end

    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset 

  # Parse individual image from a tfrecords file into TensorFlow expression.
  def _parse_tfrecord_tf(self, record, name):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True,default_value=0.0)})
        data = tf.cast(features['data'], tf.float32)
        data = tf.reshape(data, features['shape'])
        # old line
        # return {self.name:data}
        # new line
        return {name: data}


class fastMRI_dataset:
  def __init__(self,
    tfr_file_us_image,            # Path to tfrecord file.
    tfr_file_mask,
    tfr_file_coil_map,
    label_file,
    resolution=512,      # Dataset resolution.
    max_images=7500,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
    batch_size=1,
  ):
    """Adapted from https://github.com/NVlabs/stylegan2/blob/master/training/dataset.py.
    Use StyleGAN2 dataset_tool.py to generate tf record files.
    """
    self.tfr_file_us_image  = tfr_file_us_image
    self.tfr_file_mask      = tfr_file_mask
    self.tfr_file_coil_map  = tfr_file_coil_map
    self.label_file         = label_file
    self.dtype              = 'float32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb
    self.num_classes        = 6         # unconditional

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]
    self.batch_size = batch_size

  def train_input_fn(self):
    # Build TF expressions.
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    # old line
    # dset_us = dset_us.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # new line
    dset_us = dset_us.map(lambda r: self._parse_tfrecord_tf(r, 'us_im'), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_us = dset_us.take(self.max_images)


    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    # old line
    # dset_mask = dset_mask.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # new line
    dset_mask = dset_mask.map(lambda r: self._parse_tfrecord_tf(r, 'mask'), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_mask = dset_mask.take(self.max_images)

    # old line
    dset_coil_map = tf.data.TFRecordDataset(self.tfr_file_coil_map,compression_type='', buffer_size=self.buffer_mb<<20)
    # new line
    dset_coil_map = dset_coil_map.map(lambda r: self._parse_tfrecord_tf(r, 'coil_map'), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.name = 'coil_map'
    dset_coil_map = dset_coil_map.map(self._parse_tfrecord_tf_4, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_coil_map = dset_coil_map.take(self.max_images)



    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)    
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})
    _tf_labels_dataset = _tf_labels_dataset.take(self.max_images)

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask, dset_coil_map))
    dset = dset.map(lambda x,y,z : dict(us_im=x["us_im"], mask=y["mask"],  coil_map=z["coil_map"]))


    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))
    dset = dset.repeat()
    # Shuffle and prefetch
   # dset = dset.shuffle(50000)
    dset = dset.batch(self.batch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset


  def eval_input_fn(self):
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    dset_us = dset_us.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    dset_mask = dset_mask.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dset_coil_map = tf.data.TFRecordDataset(self.tfr_file_coil_map,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'coil_map'
    dset_coil_map = dset_coil_map.map(self._parse_tfrecord_tf_4, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_coil_map = dset_coil_map.take(self.max_images)

    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask, dset_coil_map))
    dset = dset.map(lambda x,y,z : dict(us_im=x["us_im"], mask=y["mask"], coil_map = z['coil_map']))
    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))
    # Shuffle and prefetch
   # dset = dset.shuffle(50000)
    dset = dset.batch(self.batch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset 


  # Parse individual image from a tfrecords file into TensorFlow expression.
  def _parse_tfrecord_tf_4(self, record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([4], tf.int64),
            'data': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True,default_value=0.0)})
        data = tf.cast(features['data'], tf.float32)
        data = tf.reshape(data, features['shape'])
        return {self.name:data}

  def _parse_tfrecord_tf_3(self, record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True,default_value=0.0)})
        data = tf.cast(features['data'], tf.float32)
        data = tf.reshape(data, features['shape'])
        return {self.name:data}


def get_dataset(name, *, batch_size=1,phase='train'):

  # Dataset locations in the original repo were hardcoded to an absolute `/data/...` path.
  # In this workspace, TFRecords live under:
  #   <repo>/data/datasets/tfrecords-datasets/
  #
  # You can override the base directory by setting:
  # SSDIFFRECON_TFRECORDS_DIR="/data/datasets/tfrecords-datasets-mod"
  # or:
  #   SSDIFFRECON_DATA_DIR=/path/to/repo-data-root   (the folder that contains `datasets/tfrecords-datasets/`)
  def _default_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    # gpu_utils/ -> diffusion_tf/ -> diffusion/ -> SSDiffRecon/ -> <repo_root>
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.join(repo_root, "data", "datasets", "tfrecords-datasets")

  tfrecords_dir = os.environ.get("SSDIFFRECON_TFRECORDS_DIR")
  data_root = os.environ.get("SSDIFFRECON_DATA_DIR")
  if tfrecords_dir:
    data_dir = tfrecords_dir
  elif data_root:
    data_dir = os.path.join(data_root, "datasets", "tfrecords-datasets")
  else:
    data_dir = _default_data_dir()
  legacy_dir = "/data/datasets/tfrecords-datasets"

  # Prefer local `data_dir` when it exists, else fall back to original absolute path.
  base = data_dir if os.path.exists(data_dir) else legacy_dir

  if name == "ixi":
    if phase == 'train':
        return IXI_dataset(
          os.path.join(base, "ixi_mixed_us_images", "-r08.tfrecords"),
          os.path.join(base, "ixi_mixed_mask_images", "-r08.tfrecords"),
          os.path.join(base, "ixi_mixed_us_images", "-rxx.labels"),
          batch_size=batch_size,
        )
    elif phase == 'test':
      return IXI_dataset(
        os.path.join(base, "ixi_mixed_test_us_images", "-r08.tfrecords"),
        os.path.join(base, "ixi_mixed_test_mask_images", "-r08.tfrecords"),
        os.path.join(base, "ixi_mixed_test_us_images", "-rxx.labels"),
        batch_size=batch_size,
      )
    elif phase == 'val':
      return IXI_dataset(
        os.path.join(base, "ixi_mixed_val_us_images", "-r08.tfrecords"),
        os.path.join(base, "ixi_mixed_val_mask_images", "-r08.tfrecords"),
        os.path.join(base, "ixi_mixed_val_us_images", "-rxx.labels"),
        batch_size=batch_size,
      )
    else:
      print("none of the phases is selected")
  
  # Custom dataset: generated images (train/test split) + original masks/labels
  elif name == "ixi_custom":
    if phase == 'train':
        return IXI_dataset(
          os.path.join(base, "ixi_custom_us_images_trial2", "-custom-train.tfrecords"),  # Generated train images
          os.path.join(base, "ixi_mixed_mask_images", "-r08.tfrecords"),         # Original train masks
          os.path.join(base, "ixi_mixed_us_images", "-rxx.labels"),              # Original train labels
          batch_size=batch_size,
        )
    elif phase == 'test':
      return IXI_dataset(
        os.path.join(base, "ixi_custom_us_images_trial2", "-custom-test.tfrecords"),        # Generated test images
        os.path.join(base, "ixi_mixed_test_mask_images", "-r08.tfrecords"),         # Original test masks
        os.path.join(base, "ixi_mixed_test_us_images", "-rxx.labels"),              # Original test labels
        batch_size=batch_size,
      )
    else:
      print("none of the phases is selected")
  
  elif name == 'fastMRI':
    if phase == 'train':
      return fastMRI_dataset(
        os.path.join(base, "fastmri_mixed_us", "-r09.tfrecords"),
        "/data/codes/codes/ssdiffrecon/datasets/fastmri_mixed_train_mask/train/train.tfrecords",
        "/data/codes/codes/ssdiffrecon/datasets/fastmri_mixed_train_coil_maps/train/train.tfrecords",
        os.path.join(base, "fastmri_mixed_us", "-rxx.labels"),
        batch_size=batch_size,
      )
    elif phase == 'val':
      return fastMRI_dataset(
        os.path.join(base, "fastmri_mixed_val_us_im", "-r09.tfrecords"),
        os.path.join(base, "fastmri_mixed_val_mask_im", "-r09.tfrecords"),
        os.path.join(base, "fastmri_mixed_val_coil_map_im", "-r09.tfrecords"),
        os.path.join(base, "fastmri_mixed_val_us_im", "-rxx.labels"),
        batch_size=batch_size,
      )
    elif phase == 'test':
      return fastMRI_dataset(
        os.path.join(base, "fastmri_mixed_test_us_im", "-r09.tfrecords"),
        "/data/codes/codes/ssdiffrecon/datasets/fastmri_mixed_test_mask/test/test.tfrecords",
        "/data/codes/codes/ssdiffrecon/datasets/fastmri_mixed_test_coil_maps/test/test.tfrecords",
        os.path.join(base, "fastmri_mixed_test_us_im", "-rxx.labels"),
        batch_size=batch_size,
      )
  else:
     print("Dataset is not defined")