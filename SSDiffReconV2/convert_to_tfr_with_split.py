"""
Convert .npy sample files to train/test tfrecords (images only).
Masks and labels will be reused from the original dataset via datasets.py configuration.

For research question: "Does training on generated data give good results?"
"""

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============ CONFIGURATION ============
# Input: Directory with .npy sample files from the original model
source_dir = './results/ixi_samples_iter1/test'

# Output directory for your custom dataset
output_dir = './data/datasets/tfrecords-datasets/ixi_custom_us_images_trial2'

# Train/test split ratio (0.8 = 80% train, 20% test)
train_ratio = 0.8
random_seed = 42

# ============ SETUP ============
os.makedirs(output_dir, exist_ok=True)


def create_tfrecord_images(file_list, output_file, is_train=True):
    """
    Create tfrecord for generated images only.
    
    Args:
        file_list: list of .npy filenames (generated samples)
        output_file: output tfrecord path
        is_train: train or test split
    """
    print(f"\nCreating {'TRAIN' if is_train else 'TEST'} image tfrecord:")
    print(f"  Output: {output_file}")
    print(f"  Samples: {len(file_list)}")
    
    with tf.io.TFRecordWriter(output_file) as writer:
        for filename in tqdm(file_list, desc=f"{'Train' if is_train else 'Test'} images"):
            # Load generated magnitude image (256, 256)
            mag_data = np.load(os.path.join(source_dir, filename)).astype(np.float32)
            
            # Check for NaN/Inf in loaded data
            if np.isnan(mag_data).any() or np.isinf(mag_data).any():
                print(f"\nWARNING: Skipping {filename} - contains NaN or Inf values")
                continue
            
            # Normalize to [0, 1] if needed
            max_val = mag_data.max()
            if max_val > 1.0 and max_val > 1e-6:  # Avoid division by zero
                mag_data = mag_data / max_val
            elif max_val < 1e-6:
                print(f"\nWARNING: Skipping {filename} - all values near zero (max={max_val})")
                continue
            
            # Clip to valid range
            mag_data = np.clip(mag_data, 0.0, 1.0)
            
            # Generate synthetic phase (necessary since samples are magnitude-only)
            phase = np.random.uniform(-np.pi, np.pi, mag_data.shape).astype(np.float32)
            real_part = mag_data * np.cos(phase)
            imag_part = mag_data * np.sin(phase)
            
            # Stack to (2, 256, 256) format
            complex_data = np.stack([real_part, imag_part], axis=0)
            
            # Final NaN check before writing
            if np.isnan(complex_data).any() or np.isinf(complex_data).any():
                print(f"\nWARNING: Skipping {filename} - NaN/Inf after processing")
                continue
            
            # Save to tfrecord
            img_flat = complex_data.flatten().tolist()
            feature = {
                'data': tf.train.Feature(float_list=tf.train.FloatList(value=img_flat)),
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=complex_data.shape))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def main():
    print("="*70)
    print("Converting Generated Samples to Train/Test TFRecords")
    print("="*70)
    print()
    
    # Get all .npy files
    all_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.npy')])
    
    if len(all_files) == 0:
        print(f"ERROR: No .npy files found in {source_dir}")
        print(f"Check that source_dir is correct: {source_dir}")
        return
    
    print(f"✓ Found {len(all_files)} generated .npy files")
    print(f"  Source: {source_dir}")
    print()
    
    # Split into train and test
    train_files, test_files = train_test_split(
        all_files, 
        train_size=train_ratio, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Split configuration:")
    print(f"  Train: {len(train_files)} samples ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"  Test:  {len(test_files)} samples ({len(test_files)/len(all_files)*100:.1f}%)")
    print()
    
    # Create tfrecords
    train_file = os.path.join(output_dir, '-custom-train.tfrecords')
    test_file = os.path.join(output_dir, '-custom-test.tfrecords')
    
    create_tfrecord_images(train_files, train_file, is_train=True)
    create_tfrecord_images(test_files, test_file, is_train=False)
    
    # Summary
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nCreated:")
    print(f"  Train images: {train_file}")
    print(f"                ({len(train_files)} samples)")
    print(f"  Test images:  {test_file}")
    print(f"                ({len(test_files)} samples)")
    print()
    print("Configuration:")
    print(f"  • Images:  GENERATED samples (from {source_dir})")
    print(f"  • Masks:   ORIGINAL dataset (ensures same undersampling pattern)")
    print(f"  • Labels:  ORIGINAL dataset (ensures same label distribution)")
    print()
    print("="*70)
    print("\nReady to train!")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
