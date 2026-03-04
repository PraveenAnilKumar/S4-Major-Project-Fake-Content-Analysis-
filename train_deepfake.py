"""
Deepfake Model Training Script for TruthGuard AI
Trains your existing .h5 models on new data to improve accuracy
"""

import numpy as np
import cv2
import os
import glob
import argparse
import logging
from datetime import datetime
from typing import Tuple, List, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepfakeTrainer:
    """
    Handles training and fine-tuning of deepfake detection models
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "datasets/deepfake"):
        """
        Initialize trainer
        
        Args:
            models_dir: Directory to save trained models
            data_dir: Directory containing training data
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.input_size = (224, 224)  # Standard input size for most models
        self.batch_size = 32
        self.epochs = 20
        self.model = None
        self.history = None
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "train/real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "train/fake"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "val/real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "val/fake"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test/real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test/fake"), exist_ok=True)
        
        logger.info(f"Trainer initialized. Models will be saved to {models_dir}")
    
    def prepare_data_generators(self, validation_split: float = 0.2) -> Tuple:
        """
        Prepare data generators with augmentation for training
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        logger.info("Preparing data generators with augmentation...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            classes=['real', 'fake']  # real=0, fake=1
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            classes=['real', 'fake']
        )
        
        logger.info(f"Found {train_generator.samples} training samples, {val_generator.samples} validation samples")
        logger.info(f"Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def prepare_test_data(self) -> tf.keras.preprocessing.image.DirectoryIterator:
        """
        Prepare test data generator (no augmentation)
        
        Returns:
            Test data generator
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            classes=['real', 'fake']
        )
        
        logger.info(f"Found {test_generator.samples} test samples")
        return test_generator
    
    def build_model(self, model_type: str = 'efficientnet', load_existing: str = None):
        """
        Build or load a model for training
        
        Args:
            model_type: Type of model to build ('efficientnet', 'mobilenet', 'resnet', 'custom')
            load_existing: Path to existing .h5 model to fine-tune
        """
        if load_existing and os.path.exists(load_existing):
            logger.info(f"Loading existing model from {load_existing} for fine-tuning...")
            self.model = keras.models.load_model(load_existing, compile=False)
            
            # Check if model needs to be recompiled with proper output layer
            if self.model.output_shape[-1] != 1:
                logger.info("Adapting model for binary classification...")
                x = self.model.layers[-2].output if len(self.model.layers) > 1 else self.model.output
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                output = layers.Dense(1, activation='sigmoid', name='new_output')(x)
                
                self.model = models.Model(inputs=self.model.input, outputs=output)
            
            logger.info("Model loaded successfully")
            return
        
        logger.info(f"Building new {model_type} model...")
        
        if model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        elif model_type == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        elif model_type == 'resnet':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        else:  # custom CNN
            self.model = self._build_custom_cnn()
            return
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=base_model.input, outputs=output)
        logger.info(f"Model built successfully with {base_model.count_params():,} base parameters")
    
    def _build_custom_cnn(self) -> models.Model:
        """Build a custom CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.input_size, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def train(self, train_generator, val_generator, fine_tune: bool = False):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            fine_tune: Whether to fine-tune all layers
        """
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = ModelCheckpoint(
            os.path.join(self.models_dir, f'deepfake_best_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Stage 1: Train only the top layers
        logger.info("Stage 1: Training top layers...")
        history1 = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // self.batch_size,
            epochs=min(10, self.epochs),
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        # Stage 2: Fine-tune all layers (if requested)
        if fine_tune:
            logger.info("Stage 2: Fine-tuning all layers...")
            
            # Unfreeze all layers
            self.model.trainable = True
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
            )
            
            history2 = self.model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // self.batch_size,
                validation_data=val_generator,
                validation_steps=val_generator.samples // self.batch_size,
                epochs=self.epochs,
                callbacks=[checkpoint, early_stop, reduce_lr],
                verbose=1
            )
            
            # Combine histories
            self.history = {}
            for key in history1.history.keys():
                self.history[key] = history1.history[key] + history2.history.get(key, [])
        else:
            self.history = history1.history
        
        # Save final model
        final_path = os.path.join(self.models_dir, f'deepfake_trained_{timestamp}.h5')
        self.model.save(final_path)
        logger.info(f"Final model saved to {final_path}")
    
    def evaluate(self, test_generator) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        # Get predictions
        y_pred_prob = self.model.predict(test_generator)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = test_generator.classes[:len(y_pred)]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Precision (Fake): {report['Fake']['precision']:.4f}")
        logger.info(f"Recall (Fake): {report['Fake']['recall']:.4f}")
        logger.info(f"F1-Score (Fake): {report['Fake']['f1-score']:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save evaluation results
        results = {
            'accuracy': float(accuracy),
            'precision_real': float(report['Real']['precision']),
            'recall_real': float(report['Real']['recall']),
            'f1_real': float(report['Real']['f1-score']),
            'precision_fake': float(report['Fake']['precision']),
            'recall_fake': float(report['Fake']['recall']),
            'f1_fake': float(report['Fake']['f1-score']),
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        results_path = os.path.join(self.models_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if not self.history:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history['loss'], label='Train Loss')
        axes[1].plot(self.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training plot saved to {save_path}")
        else:
            plt.show()
    
    def fine_tune_existing_model(self, model_path: str, train_generator, val_generator):
        """
        Fine-tune an existing .h5 model
        
        Args:
            model_path: Path to existing .h5 model
            train_generator: Training data generator
            val_generator: Validation data generator
        """
        logger.info(f"Fine-tuning existing model: {model_path}")
        self.build_model(load_existing=model_path)
        self.train(train_generator, val_generator, fine_tune=True)


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--model-type', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'resnet', 'custom'],
                       help='Type of model to train')
    parser.add_argument('--fine-tune', type=str, default=None,
                       help='Path to existing .h5 model to fine-tune')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='datasets/deepfake',
                       help='Directory containing training data')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DeepfakeTrainer(models_dir=args.models_dir, data_dir=args.data_dir)
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size
    
    # Prepare data
    train_gen, val_gen = trainer.prepare_data_generators()
    
    # Build or load model
    if args.fine_tune:
        trainer.build_model(load_existing=args.fine_tune)
        trainer.train(train_gen, val_gen, fine_tune=True)
    else:
        trainer.build_model(model_type=args.model_type)
        trainer.train(train_gen, val_gen, fine_tune=False)
    
    # Evaluate on test data
    test_gen = trainer.prepare_test_data()
    results = trainer.evaluate(test_gen)
    
    # Plot training history
    plot_path = os.path.join(args.models_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    trainer.plot_training_history(save_path=plot_path)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final accuracy: {results['accuracy']:.4f}")
    
    # Print recommendation based on results
    if results['accuracy'] > 0.95:
        logger.info("🎉 Excellent model! This will significantly improve your deepfake detection.")
    elif results['accuracy'] > 0.90:
        logger.info("✅ Good model! Should work well in production.")
    elif results['accuracy'] > 0.85:
        logger.info("👍 Decent model. Consider collecting more training data for better results.")
    else:
        logger.info("⚠️ Model needs improvement. Try collecting more diverse data or training longer.")


if __name__ == "__main__":
    main()
