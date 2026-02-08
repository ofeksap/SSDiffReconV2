import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

# Settings
source_dir = './results/ixi_iter1/test'
output_folder = './data/datasets/tfrecords-datasets/ixi_mixed_us_images_mod'
if not os.path.exists(output_folder): os.makedirs(output_folder)

record_file = os.path.join(output_folder, 'ixi-train-mod.tfrecords')

def create_tfrecord():
    files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    with tf.io.TFRecordWriter(record_file) as writer:

        for filename in tqdm(files):
            # 1. Load Magnitude (256, 256)
            mag_data = np.load(os.path.join(source_dir, filename)).astype(np.float32)

            # 2. Synthetic Phase Generation
            phase = np.random.uniform(-np.pi, np.pi, mag_data.shape).astype(np.float32)
            real_part = mag_data * np.cos(phase)
            imag_part = mag_data * np.sin(phase)

            # 3. Stack to (2, 256, 256)
            complex_data = np.stack([real_part, imag_part], axis=0)
            
            # 4. Flatten for TFRecord storage
            flat_data = complex_data.flatten().tolist()
            
            # 5. Define Feature Dictionary
            feature = {
                # 'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=complex_data.shape)),
                # 'us_im': tf.train.Feature(float_list=tf.train.FloatList(value=flat_data))
                # Save the 131,072 float values
                'data': tf.train.Feature(float_list=tf.train.FloatList(value=flat_data)),
                # Save shape as [2, 256, 256]
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=complex_data.shape))
            }
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

if __name__ == "__main__":
    create_tfrecord()
    print(f"Success! Dataset saved to {record_file}")