
#train_deepfake_batch_advanced.py


#!/usr/bin/env python
"""
train_deepfake_batch_advanced.py
Advanced batch training for deepfake detection using a memory‑efficient model.
Compatible with the TruthGuard AI ecosystem.
"""

import numpy as np
import cv2
import os
import sys
import glob
import gc
import time
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEEPFAKE DETECTOR – MEMORY‑OPTIMISED TRAINING")
print("=" * 70)

# ----------------------------------------------------------------------
# TensorFlow imports and configuration
# ----------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    print(f"✅ TensorFlow {tf.__version__} loaded")

    # GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} device(s)")
        # Mixed precision is disabled because DirectML does not support it reliably
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
        # print("✅ Mixed precision enabled")
    else:
        print("ℹ️ No GPU detected – using CPU. Training may be slow.")

except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# Dataset paths (adjust if needed)
# ----------------------------------------------------------------------
DATASET_PATH = r"D:\TruthGuard_AI_Advanced\datasets\deepfake"
TRAIN_REAL = os.path.join(DATASET_PATH, "train", "real")
TRAIN_FAKE = os.path.join(DATASET_PATH, "train", "fake")
TEST_REAL = os.path.join(DATASET_PATH, "test", "real")
TEST_FAKE = os.path.join(DATASET_PATH, "test", "fake")

for path in [TRAIN_REAL, TRAIN_FAKE, TEST_REAL, TEST_FAKE]:
    os.makedirs(path, exist_ok=True)
    print(f"✅ Directory ready: {path}")

# ----------------------------------------------------------------------
# Batch generator class (same as before, with augmentation)
# ----------------------------------------------------------------------
class AdvancedDeepfakeBatchGenerator(tf.keras.utils.Sequence):
    """Yields batches of images with optional augmentation."""

    def __init__(self, real_path, fake_path, batch_size=16, img_size=(224, 224),
                 shuffle=True, augment=False, max_samples=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment

        # Collect image paths
        self.real_paths = []
        self.fake_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.real_paths.extend(glob.glob(os.path.join(real_path, ext)))
            self.fake_paths.extend(glob.glob(os.path.join(fake_path, ext)))

        if max_samples:
            self.real_paths = self.real_paths[:max_samples]
            self.fake_paths = self.fake_paths[:max_samples]

        # Balance classes
        min_len = min(len(self.real_paths), len(self.fake_paths))
        self.real_paths = self.real_paths[:min_len]
        self.fake_paths = self.fake_paths[:min_len]

        self.labels = [0] * min_len + [1] * min_len
        self.image_paths = self.real_paths + self.fake_paths
        self.indices = np.arange(len(self.image_paths))

        print(f"  📊 Loaded {min_len} real, {min_len} fake images (balanced)")

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_idx = self.indices[start:end]

        X, y = [], []
        for i in batch_idx:
            img_path = self.image_paths[i]
            label = self.labels[i]
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype(np.float32) / 255.0
                    if self.augment:
                        img = self._augment(img)
                    X.append(img)
                    y.append(label)
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")
                continue

        # Pad batch if needed (should rarely happen)
        while len(X) < self.batch_size:
            X.append(np.zeros(self.img_size + (3,), dtype=np.float32))
            y.append(0)

        return np.array(X[:self.batch_size]), np.array(y[:self.batch_size])

    def _augment(self, img):
        """Basic augmentations – can be extended."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        # Random rotation (±15°)
        if np.random.rand() > 0.5:
            h, w = img.shape[:2]
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # Random brightness / contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)   # contrast
            beta = np.random.uniform(-0.1, 0.1)   # brightness
            img = np.clip(img * alpha + beta, 0, 1)
        return img

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ----------------------------------------------------------------------
# Model creation – using EfficientNetV2S (smaller, memory‑friendly)
# ----------------------------------------------------------------------
def create_model(input_shape=(224, 224, 3)):
    """Create an EfficientNetV2S based model."""
    base = applications.EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = True   # fine‑tune the whole model

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=out)
    return model


def count_images(directory):
    """Count images in a directory (all extensions)."""
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    total = 0
    for ext in exts:
        total += len(glob.glob(os.path.join(directory, ext)))
    return total


# ----------------------------------------------------------------------
# Main training routine
# ----------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    train_real = count_images(TRAIN_REAL)
    train_fake = count_images(TRAIN_FAKE)
    test_real = count_images(TEST_REAL)
    test_fake = count_images(TEST_FAKE)

    print(f"\nTraining:   {train_real} real, {train_fake} fake")
    print(f"Test:       {test_real} real, {test_fake} fake")

    if train_real == 0 or train_fake == 0:
        print("\n❌ Missing training images!")
        print(f"Please place images in:\n  Real: {TRAIN_REAL}\n  Fake: {TRAIN_FAKE}")
        return

    # ------------------------------------------------------------------
    # User configuration
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    try:
        batch_size = int(input("\nEnter batch size (default 8, try 4 if OOM): ") or "8")
        epochs = int(input("Enter number of epochs (default 20): ") or "20")
        max_samples = input("Max samples per class (press Enter for all): ").strip()
        max_samples = int(max_samples) if max_samples else None
    except ValueError:
        print("Invalid input, using defaults.")
        batch_size, epochs, max_samples = 8, 20, None

    # Use a smaller image size to save memory
    img_size = (224, 224)

    # ------------------------------------------------------------------
    # Build generators
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CREATING DATA GENERATORS")
    print("=" * 70)

    # First, get all paths (batch_size=1 just to collect)
    full_gen = AdvancedDeepfakeBatchGenerator(
        real_path=TRAIN_REAL,
        fake_path=TRAIN_FAKE,
        batch_size=1,
        img_size=img_size,
        shuffle=False,
        augment=False,
        max_samples=max_samples
    )
    all_paths = full_gen.image_paths
    all_labels = full_gen.labels

    # Split into train / validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    print(f"\nTraining samples:   {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    # Simple generator (without the heavy augmentation of the full class)
    class SimpleGenerator(tf.keras.utils.Sequence):
        def __init__(self, paths, labels, batch_size, img_size, augment=False):
            self.paths = paths
            self.labels = labels
            self.batch_size = batch_size
            self.img_size = img_size
            self.augment = augment
            self.indices = np.arange(len(paths))

        def __len__(self):
            return int(np.ceil(len(self.paths) / self.batch_size))

        def __getitem__(self, idx):
            start = idx * self.batch_size
            end = min(start + self.batch_size, len(self.paths))
            batch_idx = self.indices[start:end]

            X, y = [], []
            for i in batch_idx:
                img = cv2.imread(self.paths[i])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype(np.float32) / 255.0
                    if self.augment and np.random.rand() > 0.5:
                        img = np.fliplr(img).copy()
                    X.append(img)
                    y.append(self.labels[i])
            # Pad if necessary
            while len(X) < self.batch_size:
                X.append(np.zeros(self.img_size + (3,), dtype=np.float32))
                y.append(0)
            return np.array(X[:self.batch_size]), np.array(y[:self.batch_size])

        def on_epoch_end(self):
            np.random.shuffle(self.indices)

    train_gen = SimpleGenerator(train_paths, train_labels, batch_size, img_size, augment=True)
    val_gen = SimpleGenerator(val_paths, val_labels, batch_size, img_size, augment=False)

    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)
    print(f"Steps per epoch:   {steps_per_epoch}")
    print(f"Validation steps:  {validation_steps}")

    # ------------------------------------------------------------------
    # Create and compile model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = create_model(input_shape=img_size + (3,))
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
        jit_compile=True   # XLA may work; if you get errors, remove this line
    )
    model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint('models/deepfake_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)

    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # Save final model
    model.save('models/deepfake_final.h5')
    print("\n✅ Model saved to 'models/deepfake_final.h5'")

    # ------------------------------------------------------------------
    # Test evaluation (if test data exists)
    # ------------------------------------------------------------------
    if count_images(TEST_REAL) > 0 and count_images(TEST_FAKE) > 0:
        print("\n" + "=" * 70)
        print("🧪 TESTING ON TEST SET")
        print("=" * 70)

        # Collect test paths
        test_paths = []
        test_labels = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_paths.extend(glob.glob(os.path.join(TEST_REAL, ext)))
            test_paths.extend(glob.glob(os.path.join(TEST_FAKE, ext)))
            test_labels.extend([0] * len(glob.glob(os.path.join(TEST_REAL, ext))))
            test_labels.extend([1] * len(glob.glob(os.path.join(TEST_FAKE, ext))))

        test_gen = SimpleGenerator(test_paths, test_labels, batch_size, img_size, augment=False)
        test_results = model.evaluate(test_gen, verbose=1)

        print("\nTest Results:")
        print(f"  Loss:      {test_results[0]:.4f}")
        print(f"  Accuracy:  {test_results[1]:.4f}")
        if len(test_results) > 2:
            print(f"  AUC:       {test_results[2]:.4f}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)