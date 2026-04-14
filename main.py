"""
FINAL MULTI-CLASS SEGMENTATION PIPELINE (COLORED MASKS)
"""

# ============================================================================
# IMPORTS + GPU
# ============================================================================

import os, sys, cv2, zipfile, tempfile, warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("✓ Libraries imported")

# Mixed Precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("GPU:", tf.test.gpu_device_name())


# ============================================================================
# CONFIG
# ============================================================================

DATA_ZIP = "./data.zip"
OUTPUT_DIR = "./segmentation_results"
IMG_SIZE = (256,256)
NUM_CLASSES = 4  # background + 3 object types

# Delete old data
if Path(OUTPUT_DIR).exists():
    import shutil
    shutil.rmtree(OUTPUT_DIR)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# MASK ENCODING (COLOR → CLASS)
# ============================================================================

def encode_mask(mask):
    label = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Background (light gray)
    label[np.all(mask > [200,200,200], axis=-1)] = 0

    # Purple (person)
    label[(mask[:,:,0] > 100) & (mask[:,:,2] > 100)] = 1

    # Green (plants)
    label[(mask[:,:,1] > 100) & (mask[:,:,0] < 100)] = 2

    # Pink (objects)
    label[(mask[:,:,2] > 100) & (mask[:,:,1] < 100)] = 3

    return label


# ============================================================================
# PREPROCESSOR
# ============================================================================

class ServerDatasetPreprocessor:

    def __init__(self, data_zip, output_dir):
        self.data_zip = Path(data_zip)
        self.output_dir = Path(output_dir)

    def extract_zip(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(self.data_zip,'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

        self.temp_original = self.temp_dir/"input"/"original_images"
        self.temp_masks = self.temp_dir/"input"/"masked_images"

    def process_dataset(self):

        self.extract_zip()

        orig = {f.stem:f for f in self.temp_original.rglob("*") if f.suffix.lower() in ['.png','.jpg']}
        masks = {f.stem:f for f in self.temp_masks.rglob("*") if f.suffix.lower() in ['.png','.jpg']}

        keys = [k for k in orig if k in masks]

        print(f"✓ Found {len(keys)} pairs")

        train,valtest = train_test_split(keys,test_size=0.25,random_state=42)
        val,test = train_test_split(valtest,test_size=0.4,random_state=42)

        splits = {'train':train,'val':val,'test':test}

        for split,items in splits.items():
            (self.output_dir/split/'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir/split/'masks').mkdir(parents=True, exist_ok=True)

            for k in tqdm(items, desc=split):

                img = cv2.imread(str(orig[k]))
                mask = cv2.imread(str(masks[k]))

                if img is None or mask is None:
                    continue

                # Resize image
                img = cv2.resize(img, IMG_SIZE) / 255.0

                # Resize mask
                mask = cv2.resize(mask, IMG_SIZE)

                # Convert BGR → RGB
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                # Encode colors → labels
                mask = encode_mask(mask)

                # Expand dims
                mask = np.expand_dims(mask, axis=-1)

                np.save(self.output_dir/split/'images'/f"{k}.npy", img.astype(np.float32))
                np.save(self.output_dir/split/'masks'/f"{k}.npy", mask)

        print("✓ Dataset processed")


# ============================================================================
# DATALOADER
# ============================================================================

class DataLoader:
    def __init__(self, dataset_dir, batch_size=2):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size

    def create_tf_dataset(self, split='train'):

        files = sorted((self.dataset_dir/split/'images').glob("*.npy"))

        def load(x):
            x = x.numpy().decode("utf-8")
            img = np.load(x)
            mask = np.load(x.replace("images","masks"))
            return img.astype(np.float32), mask.astype(np.int32)

        ds = tf.data.Dataset.from_tensor_slices([str(f) for f in files])
        ds = ds.shuffle(1000)

        ds = ds.map(lambda x: tf.py_function(load,[x],[tf.float32,tf.int32]))

        def fix(img,mask):
            img.set_shape((256,256,3))
            mask.set_shape((256,256,1))
            return img,mask

        ds = ds.map(fix)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds


# ============================================================================
# MODEL
# ============================================================================

from tensorflow.keras import layers, models

def conv_block(x,f):
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    return x

def build_unet():
    inputs = layers.Input((256,256,3))

    c1 = conv_block(inputs,16); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1,32); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2,64); p3 = layers.MaxPooling2D()(c3)

    b = conv_block(p3,128)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1,c3])
    c4 = conv_block(u1,64)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2,c2])
    c5 = conv_block(u2,32)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3,c1])
    c6 = conv_block(u3,16)

    outputs = layers.Conv2D(NUM_CLASSES,1,activation='softmax',dtype='float32')(c6)

    return models.Model(inputs,outputs)


# ============================================================================
# COMPILE
# ============================================================================

model = build_unet()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


# ============================================================================
# RUN
# ============================================================================

pre = ServerDatasetPreprocessor(DATA_ZIP, OUTPUT_DIR)
pre.process_dataset()

loader = DataLoader(OUTPUT_DIR)

train_ds = loader.create_tf_dataset('train')
val_ds = loader.create_tf_dataset('val')
test_ds = loader.create_tf_dataset('test')

model.fit(train_ds, validation_data=val_ds, epochs=20)


# ============================================================================
# EVALUATE
# ============================================================================

model.evaluate(test_ds)
model.save(f"{OUTPUT_DIR}/model.h5")


# ============================================================================
# VISUALIZATION
# ============================================================================

for img,mask in test_ds.take(1):
    pred = model.predict(img)

    for i in range(3):
        plt.figure(figsize=(10,3))

        plt.subplot(1,3,1)
        plt.imshow(img[i])
        plt.title("Image")

        plt.subplot(1,3,2)
        plt.imshow(mask[i].squeeze())
        plt.title("Ground Truth")

        plt.subplot(1,3,3)
        pred_mask = np.argmax(pred[i], axis=-1)
        plt.imshow(pred_mask)
        plt.title("Prediction")

        plt.show()