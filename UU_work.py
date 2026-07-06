import numpy as np
from tensorflow.keras.models import load_model
# from src.model.inference_2d import HSI2DClassifier

model_path = r'C:\Users\okyer001\PycharmProjects\pythonProject\models\model.keras'

# ------ Approach 1: Raw model ------
print("Approach 1: Raw load_model")
model = load_model(model_path)
image = np.random.randn(1, 15, 15, 155).astype(np.float32)
raw_pred = model.predict(image)
print(f"Raw prediction: {raw_pred}")




print()



# """
# Complete 2D CNN Training Pipeline for HSI Classification
# ========================================================
# Input: HSI files (15x15x155 or 15x15x68) stored in nested subfolders + Excel labels
# Output: Trained Keras model + history + evaluation metrics
#
# Handles:
# - Recursive file search in nested directories
# - Automatic interpolation of 68-band files to 155 bands
# - JSON serialization of training history
# """
#
# import os
# import json
# import argparse
# from datetime import datetime
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import callbacks, optimizers, layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
#
# # Optional: MLflow for experiment tracking
# try:
#     import mlflow
#     import mlflow.tensorflow
#
#     MLFLOW_AVAILABLE = True
# except ImportError:
#     MLFLOW_AVAILABLE = False
#     print("MLflow not installed. Tracking disabled.")
#
# # Try importing tifffile; fallback to numpy if not available
# try:
#     import tifffile as tiff
#
#     TIFF_AVAILABLE = True
# except ImportError:
#     TIFF_AVAILABLE = False
#     print("Warning: tifffile not installed. Only .npy files will be supported.")
#
# # Try importing scipy for interpolation
# try:
#     from scipy.interpolate import interp1d
#
#     SCIPY_AVAILABLE = True
# except ImportError:
#     SCIPY_AVAILABLE = False
#     print("Warning: scipy not installed. 68-band files will be skipped.")
#
# print(f"TensorFlow version: {tf.__version__}")
# print(f"NumPy version: {np.__version__}")
#
#
# # ============================================================
# # 1. DATA LOADER CLASS (with recursive file search)
# # ============================================================
#
# class HSIDataLoader:
#     """
#     Loads HSI files and their labels from Excel.
#     Supports .npy, .tiff, .tif formats.
#     Recursively searches all subdirectories.
#     Handles both 155-band and 68-band files by interpolation.
#     """
#
#     def __init__(self, data_dir: str, excel_path: str):
#         """
#         Args:
#             data_dir: Root directory containing subfolders with HSI files
#             excel_path: Excel/CSV with columns: File_Name, Label (1-4)
#         """
#         self.data_dir = Path(data_dir)
#         self.excel_path = Path(excel_path)
#
#         # Class mapping: label number -> readable name
#         self.class_names = {
#             0: 'HNHP',  # High N, High P (Healthy)
#             1: 'HNLP',  # High N, Low P
#             2: 'LNHP',  # Low N, High P
#             3: 'LNLP'  # Low N, Low P (Severe stress)
#         }
#
#     def load_labels(self) -> pd.DataFrame:
#         """
#         Load labels from Excel/CSV file.
#         Uses openpyxl engine for .xlsx files.
#         """
#         if self.excel_path.suffix in ['.xlsx', '.xls']:
#             df = pd.read_excel(self.excel_path, engine='openpyxl')
#         else:
#             df = pd.read_csv(self.excel_path)
#
#         # Rename columns: assume first two columns are File_Name and Label
#         if df.shape[1] >= 2:
#             df.columns = ['File_Name', 'Label'] + list(df.columns[2:])
#         else:
#             raise ValueError(f"Excel file must have at least 2 columns. Found {df.shape[1]}")
#
#         return df
#
#     def load_images_and_labels(self) -> tuple:
#         """
#         Load all images and corresponding labels.
#         Recursively searches all subfolders for matching files.
#         Interpolates 68-band files to 155 bands if scipy is available.
#
#         Returns:
#             images: np.array of shape (n_samples, 15, 15, 155)
#             labels: np.array of shape (n_samples,) with values 0-3
#         """
#         df = self.load_labels()
#         images = []
#         labels = []
#         skipped = 0
#         found_files = 0
#         band_155_count = 0
#         band_68_count = 0
#         band_other_count = 0
#
#         print(f"Scanning {self.data_dir} and all subfolders for {len(df)} files...")
#
#         # Pre-index all files in the directory tree for faster lookup
#         file_index = {}
#         for root, dirs, files in os.walk(self.data_dir):
#             for file in files:
#                 # Store file by name without extension for quick lookup
#                 base_name = os.path.splitext(file)[0]
#                 if base_name not in file_index:
#                     file_index[base_name] = []
#                 file_index[base_name].append(os.path.join(root, file))
#
#         print(f"Found {len(file_index)} unique files in directory tree.")
#
#         # Define interpolation function if scipy is available
#         def interpolate_68_to_155(image_68):
#             """Interpolate from 68 bands to 155 bands using linear interpolation."""
#             h, w, bands = image_68.shape
#             original_indices = np.linspace(0, 1, bands)
#             target_indices = np.linspace(0, 1, 155)
#             interpolated = np.zeros((h, w, 155))
#             for i in range(h):
#                 for j in range(w):
#                     try:
#                         f = interp1d(original_indices, image_68[i, j, :],
#                                      kind='linear', fill_value='extrapolate')
#                         interpolated[i, j, :] = f(target_indices)
#                     except Exception:
#                         # Fallback: repeat the last value
#                         indices = np.linspace(0, bands - 1, 155).astype(int)
#                         interpolated[i, j, :] = image_68[i, j, indices]
#             return interpolated
#
#         for idx, row in df.iterrows():
#             filename = str(row['File_Name']).strip()
#             label = int(row['Label']) - 1  # 1-4 -> 0-3
#
#             if filename in file_index:
#                 file_path = file_index[filename][0]
#                 found_files += 1
#             else:
#                 skipped += 1
#                 continue
#
#             try:
#                 # Load based on extension
#                 if file_path.endswith('.npy'):
#                     hsi_cube = np.load(file_path).astype(np.float32)
#                 elif file_path.endswith('.npz'):
#                     data = np.load(file_path)
#                     hsi_cube = data[data.files[0]].astype(np.float32)
#                 else:
#                     if not TIFF_AVAILABLE:
#                         raise ImportError("tifffile not installed")
#                     hsi_cube = tiff.imread(file_path).astype(np.float32)
#
#                 # Ensure shape is (H, W, C) -> TensorFlow expects (H, W, C)
#                 if hsi_cube.shape[0] == 155 and len(hsi_cube.shape) == 3:
#                     # Shape is (155, 15, 15) -> transpose to (15, 15, 155)
#                     hsi_cube = np.moveaxis(hsi_cube, 0, -1)
#                 elif hsi_cube.shape == (15, 15, 155):
#                     pass  # Already correct
#                 elif hsi_cube.shape == (15, 15, 68):
#                     if SCIPY_AVAILABLE:
#                         print(f"Interpolating {filename} from 68 to 155 bands...")
#                         hsi_cube = interpolate_68_to_155(hsi_cube)
#                         band_68_count += 1
#                     else:
#                         print(f"Warning: scipy not installed. Skipping 68-band file {filename}.")
#                         skipped += 1
#                         continue
#                 else:
#                     print(f"Warning: Unexpected shape {hsi_cube.shape} for {filename}, skipping.")
#                     band_other_count += 1
#                     skipped += 1
#                     continue
#
#                 # Final check: should be (15, 15, 155)
#                 if hsi_cube.shape != (15, 15, 155):
#                     print(f"Warning: Shape {hsi_cube.shape} for {filename} (expected 15x15x155), skipping.")
#                     skipped += 1
#                     continue
#
#                 images.append(hsi_cube)
#                 labels.append(label)
#                 band_155_count += 1
#
#             except Exception as e:
#                 print(f"Warning: Could not load {filename} from {file_path}: {e}")
#                 skipped += 1
#
#         if not images:
#             print(f"Found {found_files} files but none could be loaded.")
#             print(f"Total files indexed: {len(file_index)}")
#             raise RuntimeError(f"No valid images found in {self.data_dir}.")
#
#         images = np.array(images)
#         labels = np.array(labels)
#
#         print(f"Loaded {len(images)} images")
#         print(f"  - 155-band files: {band_155_count}")
#         print(f"  - Interpolated 68-band files: {band_68_count}")
#         print(f"  - Other bands (skipped): {band_other_count}")
#         print(f"  - Files not found or skipped: {skipped}")
#         print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
#
#         return images, labels
#
#     def preprocess_snv(self, image: np.ndarray) -> np.ndarray:
#         """
#         Apply Standard Normal Variate (SNV) per pixel spectrum.
#         SNV: (x - mean) / std for each spectrum
#         """
#         mean = np.mean(image, axis=2, keepdims=True)
#         std = np.std(image, axis=2, keepdims=True) + 1e-8
#         return (image - mean) / std
#
#     def preprocess_all(self, images: np.ndarray) -> np.ndarray:
#         """Apply SNV preprocessing to all images."""
#         return np.array([self.preprocess_snv(img) for img in images])
#
#     def get_train_val_test_split(
#             self,
#             images: np.ndarray,
#             labels: np.ndarray,
#             test_size: float = 0.15,
#             val_size: float = 0.15,
#             random_state: int = 42
#     ) -> dict:
#         """
#         Split data into train, validation, test sets.
#
#         Returns:
#             dict with keys: train_images, train_labels, val_images, val_labels,
#                            test_images, test_labels
#         """
#         # First split: separate test set
#         X_train_val, X_test, y_train_val, y_test = train_test_split(
#             images, labels,
#             test_size=test_size,
#             stratify=labels,
#             random_state=random_state
#         )
#
#         # Second split: separate validation from train
#         val_ratio = val_size / (1 - test_size)
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_train_val, y_train_val,
#             test_size=val_ratio,
#             stratify=y_train_val,
#             random_state=random_state
#         )
#
#         return {
#             'train_images': X_train,
#             'train_labels': y_train,
#             'val_images': X_val,
#             'val_labels': y_val,
#             'test_images': X_test,
#             'test_labels': y_test
#         }
#
#     def create_tf_dataset(
#             self,
#             images: np.ndarray,
#             labels: np.ndarray,
#             batch_size: int = 32,
#             shuffle: bool = False,
#             augment: bool = False
#     ) -> tf.data.Dataset:
#         """
#         Create TensorFlow Dataset with one-hot labels.
#
#         Args:
#             images: Shape (n_samples, 15, 15, 155)
#             labels: Shape (n_samples,) with values 0-3
#             batch_size: Batch size
#             shuffle: Whether to shuffle
#             augment: Whether to apply data augmentation
#         Returns:
#             tf.data.Dataset
#         """
#         # One-hot encode labels (4 classes)
#         y_one_hot = tf.keras.utils.to_categorical(labels, num_classes=4)
#
#         dataset = tf.data.Dataset.from_tensor_slices((images, y_one_hot))
#
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=len(images))
#
#         dataset = dataset.batch(batch_size)
#
#         # Data augmentation
#         if augment:
#             flip_layer = tf.keras.Sequential([
#                 layers.RandomFlip("horizontal_and_vertical"),
#                 layers.RandomRotation(0.05)  # Small rotations
#             ])
#             dataset = dataset.map(
#                 lambda x, y: (flip_layer(x, training=True), y),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )
#
#         return dataset.prefetch(tf.data.AUTOTUNE)
#
#
# # ============================================================
# # 2. MODEL ARCHITECTURES
# # ============================================================
#
# def Shallow2DCNN(input_shape=(15, 15, 155), num_classes=4):
#     """Simple 2D CNN with few layers - fastest training."""
#     return models.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#
#
# def DepthwiseSeparableCNN(input_shape=(15, 15, 155), num_classes=4):
#     """Efficient CNN using depthwise separable convolutions."""
#     return models.Sequential([
#         layers.Input(shape=input_shape),
#         layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#
#
# def SpectralSpatialCNN(input_shape=(15, 15, 155), num_classes=4):
#     """
#     Spectral-spatial CNN with 1x1 convolutions for band reduction
#     and 3x3 convolutions for spatial features.
#     """
#     inputs = layers.Input(shape=input_shape)
#
#     # Spectral reduction (1x1 conv)
#     x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
#     x = layers.BatchNormalization()(x)
#
#     # Spatial feature extraction (3x3 conv)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#
#     # More spatial features
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.GlobalAveragePooling2D()(x)
#
#     # Classification
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dropout(0.3)(x)
#     outputs = layers.Dense(num_classes, activation='softmax')(x)
#
#     return models.Model(inputs=inputs, outputs=outputs)
#
#
# # ============================================================
# # 3. TRAINING FUNCTION
# # ============================================================
#
# def train_model(
#         data_dir: str,
#         label_file: str,
#         model_type: str = 'spectral_spatial',
#         batch_size: int = 32,
#         epochs: int = 100,
#         learning_rate: float = 1e-3,
#         model_save_path: str = './models/model.keras',
#         log_dir: str = './logs',
#         use_augmentation: bool = True,
#         test_size: float = 0.15,
#         val_size: float = 0.15,
#         use_mlflow: bool = False
# ) -> tuple:
#     """
#     Main training function.
#
#     Returns:
#         model: Trained Keras model
#         history: Training history
#         test_metrics: Dictionary of test metrics
#     """
#
#     # Create directories
#     os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
#
#     # ========== 1. Load Data ==========
#     print("=" * 60)
#     print("STEP 1: Loading Data")
#     print("=" * 60)
#
#     loader = HSIDataLoader(data_dir, label_file)
#     images, labels = loader.load_images_and_labels()
#
#     print(f"Dataset shape: {images.shape}")
#     print(f"Labels shape: {labels.shape}")
#
#     # ========== 2. Preprocess ==========
#     print("\n" + "=" * 60)
#     print("STEP 2: Preprocessing (SNV Normalization)")
#     print("=" * 60)
#
#     images = loader.preprocess_all(images)
#     print("Preprocessing complete.")
#
#     # ========== 3. Split Data ==========
#     print("\n" + "=" * 60)
#     print("STEP 3: Splitting Data")
#     print("=" * 60)
#
#     split = loader.get_train_val_test_split(images, labels, test_size, val_size)
#
#     X_train, y_train = split['train_images'], split['train_labels']
#     X_val, y_val = split['val_images'], split['val_labels']
#     X_test, y_test = split['test_images'], split['test_labels']
#
#     print(f"Training set: {X_train.shape[0]} samples")
#     print(f"Validation set: {X_val.shape[0]} samples")
#     print(f"Test set: {X_test.shape[0]} samples")
#
#     # Print class distribution
#     for name, y_set in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
#         unique, counts = np.unique(y_set, return_counts=True)
#         print(f"{name} class distribution: {dict(zip(unique, counts))}")
#
#     # ========== 4. Create Datasets ==========
#     print("\n" + "=" * 60)
#     print("STEP 4: Creating TensorFlow Datasets")
#     print("=" * 60)
#
#     train_ds = loader.create_tf_dataset(
#         X_train, y_train, batch_size,
#         shuffle=True, augment=use_augmentation
#     )
#     val_ds = loader.create_tf_dataset(
#         X_val, y_val, batch_size,
#         shuffle=False, augment=False
#     )
#     test_ds = loader.create_tf_dataset(
#         X_test, y_test, batch_size,
#         shuffle=False, augment=False
#     )
#
#     print(f"Training batches: {len(train_ds)}")
#     print(f"Validation batches: {len(val_ds)}")
#     print(f"Test batches: {len(test_ds)}")
#
#     # ========== 5. Build Model ==========
#     print("\n" + "=" * 60)
#     print(f"STEP 5: Building Model ({model_type})")
#     print("=" * 60)
#
#     input_shape = (15, 15, 155)
#     num_classes = 4
#
#     model_constructors = {
#         'spectral_spatial': SpectralSpatialCNN,
#         'shallow': Shallow2DCNN,
#         'depthwise': DepthwiseSeparableCNN
#     }
#
#     if model_type not in model_constructors:
#         raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(model_constructors.keys())}")
#
#     model = model_constructors[model_type](input_shape, num_classes)
#
#     # Compile
#     optimizer = optimizers.Adam(learning_rate=learning_rate)
#     model.compile(
#         optimizer=optimizer,
#         loss='categorical_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall')
#         ]
#     )
#
#     model.summary()
#
#     # ========== 6. Setup Callbacks ==========
#     print("\n" + "=" * 60)
#     print("STEP 6: Setting Up Callbacks")
#     print("=" * 60)
#
#     callbacks_list = [
#         callbacks.EarlyStopping(
#             monitor='val_accuracy',
#             patience=20,
#             restore_best_weights=True,
#             verbose=1
#         ),
#         callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=10,
#             min_lr=1e-7,
#             verbose=1
#         ),
#         callbacks.ModelCheckpoint(
#             filepath=model_save_path,
#             monitor='val_accuracy',
#             save_best_only=True,
#             verbose=1
#         ),
#         callbacks.TensorBoard(
#             log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
#         )
#     ]
#
#     # ========== 7. Train ==========
#     print("\n" + "=" * 60)
#     print("STEP 7: Training")
#     print("=" * 60)
#
#     # MLflow tracking (optional)
#     if use_mlflow and MLFLOW_AVAILABLE:
#         mlflow.tensorflow.autolog()
#         mlflow.start_run()
#         mlflow.log_params({
#             "model_type": model_type,
#             "batch_size": batch_size,
#             "epochs": epochs,
#             "learning_rate": learning_rate,
#             "use_augmentation": use_augmentation,
#             "train_samples": len(X_train),
#             "val_samples": len(X_val),
#             "test_samples": len(X_test)
#         })
#
#     history = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=epochs,
#         callbacks=callbacks_list,
#         verbose=1
#     )
#
#     # ========== 8. Evaluate ==========
#     print("\n" + "=" * 60)
#     print("STEP 8: Evaluation on Test Set")
#     print("=" * 60)
#
#     test_results = model.evaluate(test_ds, verbose=1)
#     test_metrics = dict(zip(model.metrics_names, test_results))
#
#     print("Test Metrics:")
#     for metric, value in test_metrics.items():
#         print(f"  {metric}: {value:.4f}")
#
#     # Classification report
#     y_true = []
#     y_pred = []
#     for batch_x, batch_y in test_ds:
#         preds = model.predict(batch_x, verbose=0)
#         y_true.extend(np.argmax(batch_y.numpy(), axis=1))
#         y_pred.extend(np.argmax(preds, axis=1))
#
#     class_names = ['HNHP', 'HNLP', 'LNHP', 'LNLP']
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=class_names))
#
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_true, y_pred))
#
#     if use_mlflow and MLFLOW_AVAILABLE:
#         mlflow.log_metrics(test_metrics)
#         mlflow.end_run()
#
#     # ========== 9. Save Model and History ==========
#     print("\n" + "=" * 60)
#     print("STEP 9: Saving Model and History")
#     print("=" * 60)
#
#     # Save final model
#     model.save(model_save_path)
#     print(f"Model saved to: {model_save_path}")
#
#     # Helper function to convert numpy types to JSON serializable
#     def convert_to_serializable(obj):
#         """Recursively convert numpy types to Python types for JSON serialization."""
#         if isinstance(obj, (np.float32, np.float64)):
#             return float(obj)
#         elif isinstance(obj, (np.int32, np.int64)):
#             return int(obj)
#         elif isinstance(obj, dict):
#             return {k: convert_to_serializable(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_to_serializable(item) for item in obj]
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return obj
#
#     # Save training history
#     history_path = model_save_path.replace('.keras', '_history.json')
#     history_serializable = convert_to_serializable(history.history)
#     with open(history_path, 'w') as f:
#         json.dump(history_serializable, f, indent=2)
#     print(f"Training history saved to: {history_path}")
#
#     # Save test metrics
#     metrics_path = model_save_path.replace('.keras', '_test_metrics.json')
#     metrics_serializable = convert_to_serializable(test_metrics)
#     with open(metrics_path, 'w') as f:
#         json.dump(metrics_serializable, f, indent=2)
#     print(f"Test metrics saved to: {metrics_path}")
#
#     # Save model summary
#     summary_path = model_save_path.replace('.keras', '_summary.txt')
#     with open(summary_path, 'w') as f:
#         model.summary(print_fn=lambda x: f.write(x + '\n'))
#     print(f"Model summary saved to: {summary_path}")
#
#     print("\n" + "=" * 60)
#     print("TRAINING COMPLETE!")
#     print("=" * 60)
#
#     return model, history, test_metrics
#
#
# # ============================================================
# # 4. COMMAND LINE INTERFACE
# # ============================================================
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Train 2D CNN for Hyperspectral Image Classification',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Train with default settings (using hardcoded defaults if you set them)
#   python UU_work.py
#
#   # Override paths and model type
#   python UU_work.py --data_dir D:/MyData --label_file labels.xlsx --model_type depthwise
#         """
#     )
#
#     # Defaults for your system (set to your paths)
#     parser.add_argument('--data_dir',
#                         default=r'C:\Frank\New folder\Quinoa_dataset-hyper',
#                         help='Root directory containing subfolders with HSI files')
#     parser.add_argument('--label_file',
#                         default=r'C:\Frank\New folder\Quinoa_dataset-hyper\quinoa_hsi_labels.xlsx',
#                         help='Excel/CSV file with File_Name and Label columns')
#     parser.add_argument('--model_type', default='spectral_spatial',
#                         choices=['spectral_spatial', 'shallow', 'depthwise'],
#                         help='Model architecture to use')
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help='Batch size for training')
#     parser.add_argument('--epochs', type=int, default=100,
#                         help='Maximum number of epochs')
#     parser.add_argument('--learning_rate', type=float, default=1e-3,
#                         help='Initial learning rate')
#     parser.add_argument('--output', default='./models/model.keras',
#                         help='Output path for the trained model')
#     parser.add_argument('--log_dir', default='./logs',
#                         help='Directory for TensorBoard logs')
#     parser.add_argument('--test_size', type=float, default=0.15,
#                         help='Proportion of data for test set')
#     parser.add_argument('--val_size', type=float, default=0.15,
#                         help='Proportion of data for validation set')
#     parser.add_argument('--no_augment', action='store_true',
#                         help='Disable data augmentation')
#     parser.add_argument('--use_mlflow', action='store_true',
#                         help='Enable MLflow tracking')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='Random seed for reproducibility')
#
#     args = parser.parse_args()
#
#     # Set random seeds for reproducibility
#     np.random.seed(args.seed)
#     tf.random.set_seed(args.seed)
#
#     # Run training
#     train_model(
#         data_dir=args.data_dir,
#         label_file=args.label_file,
#         model_type=args.model_type,
#         batch_size=args.batch_size,
#         epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         model_save_path=args.output,
#         log_dir=args.log_dir,
#         use_augmentation=not args.no_augment,
#         test_size=args.test_size,
#         val_size=args.val_size,
#         use_mlflow=args.use_mlflow
#     )




























# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   SEGMENTAITON-- USING NDVI+ AND CROPPING USING GRID CLICK
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import json
import cv2
import numpy as np
from skimage import morphology
import tifffile
import matplotlib.pyplot as plt
from spectral import open_image
import gc

# ==========================================================
# CONFIGURABLE BANDS
# ==========================================================
# For the ORIGINAL pseudo-RGB visualisation (pre‑segmentation)
# Option A: Three different red/red‑edge bands (false colour)
# Adjust these indices to match your sensor's red wavelengths
VIS_RGB_BANDS = (150, 80, 50)   # example: three red‑region bands
# Option B: Use the same red band for all three channels (greyscale)
# VIS_RGB_BANDS = (85, 85, 85)   # replace 85 with your chosen red band index

# Bands used for segmentation (RGB preview after masking)
RGB_BANDS = (85, 59, 17)        # standard RGB for segmented preview
NDVI_BANDS = (163, 85, 85)      # (NIR, Red) for NDVI calculation

# ==========================================================
# LOAD MAPPING
# ==========================================================
with open('plotname_to_lkid_new.json', 'r') as f:
    PLOTNAME_TO_LKID = json.load(f)

# ==========================================================
# GLOBALS FOR INTERACTIVE CROPPING
# ==========================================================
refPt = []
cropping = False
image_display = None
seg_cube = None
rgb_preview_bgr = None
clone = None
grid_rows = 5
grid_cols = 5

def draw_grid(img, x1, y1, x2, y2, rows, cols):
    img_copy = img.copy()
    w, h = x2 - x1, y2 - y1
    cw, ch = w / cols, h / rows
    for i in range(1, cols):
        cx = int(x1 + i * cw)
        cv2.line(img_copy, (cx, y1), (cx, y2), (0, 255, 255), 1)
    for i in range(1, rows):
        cy = int(y1 + i * ch)
        cv2.line(img_copy, (x1, cy), (x2, cy), (0, 255, 255), 1)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, image_display, clone, grid_rows, grid_cols
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        if len(refPt) == 2:
            x1, y1 = refPt[0]
            x2, y2 = refPt[1]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            preview = draw_grid(clone, x1, y1, x2, y2, grid_rows, grid_cols)
            image_display[:] = preview
            cv2.imshow("Grid drawing", image_display)

def save_grid_coordinates(output_dir, base_name, grid_boxes, img_shape, rows, cols):
    coords = {"image_shape": [int(img_shape[1]), int(img_shape[0])], "grid_rows": rows, "grid_cols": cols, "cells": []}
    for idx, box in enumerate(grid_boxes):
        r, c = divmod(idx, cols)
        coords["cells"].append({"cell_index": idx, "row": r, "col": c,
                                "top_left": [box[0], box[1]], "bottom_right": [box[2], box[3]]})
    json_path = os.path.join(output_dir, f"{base_name}_coords.json")
    with open(json_path, 'w') as f:
        json.dump(coords, f, indent=4)
    print(f"Saved grid coordinates to {json_path}")
    return json_path

def interactive_grid_crop(seg_cube, rgb_preview_bgr, output_dir, base_name, rows, cols):
    print(f"[DEBUG] rows = {rows}, cols = {cols}")
    global image_display, clone, refPt, grid_rows, grid_cols
    grid_rows, grid_cols = rows, cols
    image_display = rgb_preview_bgr.copy()
    clone = image_display.copy()
    refPt = []
    cv2.namedWindow("Grid drawing", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Grid drawing", click_and_crop)
    print("\nDraw rectangle over tray, then press:")
    print("  'g' -> crop and save")
    print("  'r' -> reset")
    print("  'q' -> skip this file")

    # Special grid: 5 rows and 5 columns
    is_special = (rows == 5 and cols == 5)
    offset = 5 if is_special else 0   # dummy indices 0..4 (5 files), real start at 5

    while True:
        cv2.imshow("Grid drawing", image_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            image_display = clone.copy()
            refPt = []
        elif key == ord('g'):
            if len(refPt) == 2:
                x1, y1 = refPt[0]
                x2, y2 = refPt[1]
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                cell_w = (x2 - x1) / cols
                cell_h = (y2 - y1) / rows
                boxes = []
                for r in range(rows):
                    for c in range(cols):
                        left = int(x1 + c * cell_w)
                        right = int(x1 + (c+1) * cell_w)
                        top = int(y1 + r * cell_h)
                        bottom = int(y1 + (r+1) * cell_h)
                        boxes.append([left, top, right, bottom])

                crops_dir = os.path.join(output_dir, base_name)
                os.makedirs(crops_dir, exist_ok=True)

                if is_special:
                    dummy_h = boxes[0][3] - boxes[0][1]
                    dummy_w = boxes[0][2] - boxes[0][0]
                    dummy_cube = np.zeros((dummy_h, dummy_w, seg_cube.shape[2]), dtype=np.float32)
                    dummy_png = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
                    for idx in range(offset):   # 0,1,2,3,4
                        dummy_tiff_path = os.path.join(crops_dir, f"{base_name}_{idx:03d}.tiff")
                        tifffile.imwrite(dummy_tiff_path, dummy_cube)
                        dummy_png_path = os.path.join(crops_dir, f"{base_name}_{idx:03d}.png")
                        cv2.imwrite(dummy_png_path, dummy_png)
                        print(f"  Created dummy file {idx:03d}")

                for idx, (l, t, r_, b) in enumerate(boxes):
                    actual_idx = idx + offset
                    crop_cube = seg_cube[t:b, l:r_, :].copy()
                    tiff_path = os.path.join(crops_dir, f"{base_name}_{actual_idx:03d}.tiff")
                    tifffile.imwrite(tiff_path, crop_cube.astype(np.float32))
                    crop_rgb = rgb_preview_bgr[t:b, l:r_, :].copy()
                    png_path = os.path.join(crops_dir, f"{base_name}_{actual_idx:03d}.png")
                    cv2.imwrite(png_path, crop_rgb)
                    print(f"  Saved actual crop {actual_idx:03d}")

                save_grid_coordinates(output_dir, base_name, boxes, seg_cube.shape, rows, cols)
                print(f"Saved {len(boxes)} actual crops to {crops_dir}")
                if is_special:
                    print(f"Plus {offset} dummy files (indices 000-{offset-1:03d})")
                cv2.destroyWindow("Grid drawing")
                return True
            else:
                print("No rectangle drawn")
        elif key == ord('q'):
            cv2.destroyWindow("Grid drawing")
            return False
    return False

# ==========================================================
# SEGMENTATION (with configurable original RGB bands)
# ==========================================================
def process_hdr(hdr_path, output_dir, save_mask=True):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing: {hdr_path}")

    img = open_image(hdr_path)
    hsi = img.load()
    im = hsi[:, :, :224].astype(np.float32)
    del img, hsi
    gc.collect()

    # ---------- Save original pseudo-RGB using VIS_RGB_BANDS ----------
    # Extract the three bands (could be all red bands)
    rgb_original = im[:, :, VIS_RGB_BANDS].copy()
    # Stretch each band to 0-255 for visualisation
    rgb_stretched = np.zeros_like(rgb_original)
    for i in range(3):
        band = rgb_original[:, :, i]
        minv, maxv = band.min(), band.max()
        if maxv > minv:
            rgb_stretched[:, :, i] = 255 * (band - minv) / (maxv - minv)
        else:
            rgb_stretched[:, :, i] = 0
    rgb_display = rgb_stretched.astype(np.uint8)
    # Save original RGB preview
    base_hdr_name = os.path.splitext(os.path.basename(hdr_path))[0]
    original_rgb_path = os.path.join(output_dir, f"{base_hdr_name}_original_rgb.png")
    plt.imsave(original_rgb_path, rgb_display)
    print(f"Saved original pseudo-RGB (bands {VIS_RGB_BANDS}): {original_rgb_path}")
    # ------------------------------------------------------------------

    # Continue with segmentation using RGB_BANDS and NDVI_BANDS
    R = im[:, :, RGB_BANDS[0]]
    G = im[:, :, RGB_BANDS[1]]
    B = im[:, :, RGB_BANDS[2]]
    IR = im[:, :, NDVI_BANDS[0]]
    Red = im[:, :, NDVI_BANDS[1]]

    NDVI = (IR - Red) / (IR + Red + 1e-6)
    del IR, Red
    NDVI_norm = cv2.normalize(NDVI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_ndvi = cv2.threshold(NDVI_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    del NDVI, NDVI_norm

    ExG = (2 * G) - R - B
    ExGR = ExG - ((1.4 * R) - G)
    del R, G, B, ExG
    ExGR_norm = cv2.normalize(ExGR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_exgr = cv2.threshold(ExGR_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    del ExGR, ExGR_norm

    combined_mask = cv2.bitwise_and(mask_ndvi, mask_exgr)
    del mask_ndvi, mask_exgr
    # del mask_exgr

    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    # cleaned = cv2.morphologyEx(mask_ndvi, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = morphology.remove_small_objects(cleaned.astype(bool), min_size=500)
    mask = cleaned.astype(np.uint8) * 255
    del combined_mask, kernel, cleaned

    seg_cube = im * (mask[:, :, np.newaxis] / 255.0)
    del im
    seg_cube = cv2.normalize(seg_cube, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    masked_rgb = seg_cube[:, :, RGB_BANDS]
    masked_rgb = np.clip(masked_rgb * 255, 0, 255).astype(np.uint8)
    masked_rgb[mask == 0] = [255, 255, 255]

    preview_path = os.path.join(output_dir, f"{base_hdr_name}_segmented_rgb.png")
    plt.imsave(preview_path, masked_rgb)
    print(f"Saved segmented RGB preview: {preview_path}")

    rgb_preview_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)

    if save_mask:
        mask_path = os.path.join(output_dir, f"{base_hdr_name}_mask.png")
        cv2.imwrite(mask_path, mask)

    file = os.path.basename(hdr_path)
    plot_code = file[20:25] if len(file) > 25 else "unknown"
    lk_id = PLOTNAME_TO_LKID.get(plot_code, plot_code)
    plot_number = file[15:25] if len(file) > 25 else "unknown"
    date_code = file[26:34] if len(file) > 34 else "unknown"
    base_name = f"{lk_id}_{plot_number}_{date_code}"

    return seg_cube, rgb_preview_bgr, mask, base_name

# ==========================================================
# MAIN – process each .hdr file independently
# ==========================================================
if __name__ == "__main__":
    root_folder = r'E:\Utrecht U-files\New folder\New folder'
    output_root = r"C:\Frank\Segmented\New folder (2)"
    GRID_ROWS = 6          # 5x5 grid → dummy files 000-004, real start at 005
    GRID_COLS = 5

    for root, dirs, files in os.walk(root_folder):
        hdr_files = [f for f in files if f.endswith(".hdr")]
        if not hdr_files:
            continue

        for hdr_file in hdr_files:
            hdr_path = os.path.join(root, hdr_file)
            containing_folder = os.path.basename(root)
            out_dir = os.path.join(output_root, containing_folder)

            print(f"\n{'='*50}")
            print(f"Processing: {hdr_path}")
            print(f"{'='*50}")

            seg_cube, rgb_preview_bgr, mask, base_name = process_hdr(hdr_path, out_dir, save_mask=True)
            success = interactive_grid_crop(seg_cube, rgb_preview_bgr, out_dir, base_name, GRID_ROWS, GRID_COLS)
            if not success:
                print(f"Skipped {hdr_file}")

            del seg_cube, rgb_preview_bgr, mask
            gc.collect()

    print("\nAll done.")

print("\nAll done.")




import os
import json
import cv2
import numpy as np
from skimage import measure
from skimage import morphology
import tifffile
import matplotlib.pyplot as plt
from spectral import open_image
import gc

# ==========================================================
# LOAD MAPPING
# ==========================================================

with open('plotname_to_lkid_new.json', 'r') as f:
    PLOTNAME_TO_LKID = json.load(f)

# ==========================================================
# MAIN FUNCTION
# ==========================================================

def segment_hsi_by_exg_ndvi(
        hdr_path,
        output_base_dir,          # top-level output folder
        subfolder_name,           # name of the subfolder to create under output_base_dir
        save_mask=False):

    try:
        # Create the per‑folder output directory
        output_dir = os.path.join(output_base_dir, subfolder_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nLoading HSI: {hdr_path}")
        img = open_image(hdr_path)
        hsi = img.load()
        im = hsi[:, :, :224].astype(np.float32)
        del img, hsi
        gc.collect()

        rgb_bands = (85, 59, 17)
        ndvi_bands = (163, 85, 85)
        rgb_rgb = (150, 100, 57)

        R = im[:, :, rgb_bands[0]]
        G = im[:, :, rgb_bands[1]]
        B = im[:, :, rgb_bands[2]]
        IR = im[:, :, ndvi_bands[0]]
        Red = im[:, :, ndvi_bands[1]]

        NDVI = (IR - Red) / (IR + Red + 1e-6)
        del IR, Red
        NDVI_norm = cv2.normalize(NDVI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_ndvi = cv2.threshold(NDVI_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        del NDVI, NDVI_norm

        ExG = (2 * G) - R - B
        ExGR = ExG - ((1.4 * R) - G)
        del R, G, B, ExG
        ExGR_norm = cv2.normalize(ExGR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_exgr = cv2.threshold(ExGR_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        del ExGR, ExGR_norm

        combined_mask = cv2.bitwise_and(mask_ndvi, mask_exgr)
        del mask_ndvi, mask_exgr

        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = morphology.remove_small_objects(cleaned_mask.astype(bool), min_size=500)
        mask = cleaned_mask.astype(np.uint8) * 255
        del combined_mask, kernel, cleaned_mask

        seg_bgr = im * (mask[:, :, np.newaxis] / 255.0)
        del im
        seg_bgr = cv2.normalize(seg_bgr, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        image_height, image_width = seg_bgr.shape[0], seg_bgr.shape[1]
        print(f"\nWhole image shape: {image_height} x {image_width}")

        masked_rgb = seg_bgr[:, :, rgb_bands]
        masked_rgb = np.clip(masked_rgb * 255, 0, 255).astype(np.uint8)
        masked_rgb[mask == 0] = [255, 255, 255]

        base_name = os.path.splitext(os.path.basename(hdr_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_segmented_rgb.png")
        plt.imsave(out_path, masked_rgb)
        print(f"Saved segmented RGB: {out_path}")
        del masked_rgb

        if save_mask:
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask)
            print(f"Saving binary mask: {mask_path}")

        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        min_area = 1000
        region_data = []
        for region in regions:
            if region.area < min_area:
                continue
            cy, cx = region.centroid
            minr, minc, maxr, maxc = region.bbox
            region_data.append({
                "region": region,
                "cy": cy, "cx": cx,
                "minr": minr, "minc": minc,
                "maxr": maxr, "maxc": maxc
            })

        region_data = sorted(region_data, key=lambda x: x["cy"])
        row_threshold = 80
        rows = []
        for item in region_data:
            placed = False
            for row in rows:
                if abs(item["cy"] - row[0]["cy"]) < row_threshold:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])
        sorted_regions = []
        for row in rows:
            row = sorted(row, key=lambda x: x["cx"])
            sorted_regions.extend(row)

        print("\nCrop order:\n")
        for idx, item in enumerate(sorted_regions):
            print(f"{idx:03d} | top={int(item['minr'])} left={int(item['minc'])}")

        file = os.path.basename(hdr_path)
        plot_code = file[20:25]
        lk_id = PLOTNAME_TO_LKID.get(plot_code)
        if lk_id is None:
            print(f"Skipping file (plot code not found): {file}")
            return
        plot_number = file[15:25]
        date_code = file[26:34]
        new_name = f"{lk_id}_{plot_number}_{date_code}"
        crop_dir = os.path.join(output_dir, f"{new_name}")
        os.makedirs(crop_dir, exist_ok=True)

        crop_coordinates = {}
        saved_count = 0
        margin = 10
        for item in sorted_regions:
            minr = max(item["minr"] - margin, 0)
            minc = max(item["minc"] - margin, 0)
            maxr = min(item["maxr"] + margin, image_height)
            maxc = min(item["maxc"] + margin, image_width)

            crop_name = f"{new_name}_{saved_count:03d}"
            crop_coordinates[crop_name] = {
                "top_left": (int(minr), int(minc)),
                "bottom_right": (int(maxr), int(maxc)),
                "image_shape": (image_height, image_width)
            }
            print(f"\n{crop_name}\nCoordinates = [({minr}, {minc}), ({maxr}, {maxc})]")

            crop = seg_bgr[minr:maxr, minc:maxc, :].copy()
            crop_path = os.path.join(crop_dir, crop_name + ".tiff")
            tifffile.imwrite(crop_path, crop.astype(np.float32))
            del crop

            crop_rgb = seg_bgr[minr:maxr, minc:maxc, rgb_rgb].copy()
            crop_rgb = np.clip(crop_rgb * 255, 0, 255).astype(np.uint8)
            crop_mask = mask[minr:maxr, minc:maxc]
            crop_rgb[crop_mask == 0] = [255, 255, 255]
            rgb_path = os.path.join(crop_dir, crop_name + ".png")
            plt.imsave(rgb_path, crop_rgb)
            del crop_rgb, crop_mask
            print(f"Saved crop {saved_count:03d}")
            saved_count += 1

        coordinate_path = os.path.join(crop_dir, f"{new_name}_crop_coordinates.json")
        with open(coordinate_path, "w") as f:
            json.dump(crop_coordinates, f, indent=4)
        print(f"\nSaved crop coordinates: {coordinate_path}")
        print(f"\nSaved {saved_count} cropped plants to: {crop_dir}")

        del seg_bgr, mask, labeled_mask
        gc.collect()

    except Exception as e:
        print(f"Error processing {hdr_path}: {str(e)}")
    finally:
        gc.collect()

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    root_folder = r"D:\Hyperspec\Grouped_IL_New\New folder"
    output_root = r"E:\Utrecht U-files\Segmented\Main"

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".hdr"):
                hdr_path = os.path.join(root, file)
                # Get the folder that directly contains the .hdr file
                containing_folder = os.path.basename(os.path.dirname(hdr_path))
                # Use that folder name as the subfolder under output_root
                segment_hsi_by_exg_ndvi(
                    hdr_path,
                    output_root,
                    containing_folder,
                    save_mask=True
                )
                gc.collect()

print("\nDONE")






print("\nDONE")


# ========================================================================
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import json
import cv2
import numpy as np
from skimage import measure
from skimage import morphology
import tifffile
import matplotlib.pyplot as plt
from spectral import open_image
import gc

# ==========================================================
# LOAD MAPPING
# ==========================================================

with open('plotname_to_lkid_new.json', 'r') as f:
    PLOTNAME_TO_LKID = json.load(f)


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def segment_hsi_by_exg_ndvi(hdr_path, output_dir, save_mask=False):

    try:

        os.makedirs(output_dir, exist_ok=True)

        # --------------------------------------------------
        # LOAD HSI
        # --------------------------------------------------

        print(f"Loading HSI: {hdr_path}")

        img = open_image(hdr_path)
        hsi = img.load()
        im = hsi[:, :, :224].astype(np.float32)
        del img, hsi
        gc.collect()
        # --------------------------------------------------
        # BAND INDICES
        # --------------------------------------------------

        rgb_bands = (85, 59, 17)
        ndvi_bands = (163, 85, 85)
        rgb_rgb = (150, 100, 57)

        # --------------------------------------------------
        # EXTRACT BANDS
        # --------------------------------------------------

        R = im[:, :, rgb_bands[0]]
        G = im[:, :, rgb_bands[1]]
        B = im[:, :, rgb_bands[2]]
        IR = im[:, :, ndvi_bands[0]]
        Red = im[:, :, ndvi_bands[1]]

        # --------------------------------------------------
        # NDVI
        # --------------------------------------------------

        NDVI = (IR - Red) / (IR + Red + 1e-6)
        del IR, Red
        NDVI_norm = cv2.normalize(
            NDVI,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        _, mask_ndvi = cv2.threshold(
            NDVI_norm,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        del NDVI, NDVI_norm

        # --------------------------------------------------
        # ExGR
        # --------------------------------------------------

        ExG = (2 * G) - R - B
        ExGR = ExG - ((1.4 * R) - G)
        del R, G, B, ExG
        ExGR_norm = cv2.normalize(
            ExGR,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        _, mask_exgr = cv2.threshold(
            ExGR_norm,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        del ExGR, ExGR_norm

        # --------------------------------------------------
        # COMBINE MASKS
        # --------------------------------------------------

        combined_mask = cv2.bitwise_and(
            mask_ndvi,
            mask_exgr
        )

        del mask_ndvi, mask_exgr

        # --------------------------------------------------
        # MORPHOLOGICAL CLEANING
        # --------------------------------------------------

        kernel = np.ones((3, 3), np.uint8)

        cleaned_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        cleaned_mask = cv2.morphologyEx(
            cleaned_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        cleaned_mask = morphology.remove_small_objects(
            cleaned_mask.astype(bool),
            min_size=500
        )

        mask = cleaned_mask.astype(np.uint8) * 255

        del combined_mask
        del kernel
        del cleaned_mask

        # --------------------------------------------------
        # APPLY MASK
        # --------------------------------------------------

        seg_bgr = im * (
            mask[:, :, np.newaxis] / 255.0
        )

        del im

        seg_bgr = cv2.normalize(
            seg_bgr,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )

        # --------------------------------------------------
        # RGB PREVIEW
        # --------------------------------------------------

        masked_rgb = seg_bgr[:, :, rgb_bands]

        masked_rgb = np.clip(
            masked_rgb * 255,
            0,
            255
        ).astype(np.uint8)

        masked_rgb[mask == 0] = [255, 255, 255]

        # --------------------------------------------------
        # SAVE SEGMENTED RGB
        # --------------------------------------------------

        base_name = os.path.splitext(
            os.path.basename(hdr_path)
        )[0]

        out_path = os.path.join(
            output_dir,
            f"{base_name}_segmented_rgb.png"
        )

        plt.imsave(out_path, masked_rgb)

        print(f"Saved segmented RGB: {out_path}")

        del masked_rgb

        # --------------------------------------------------
        # SAVE MASK
        # --------------------------------------------------

        if save_mask:

            mask_path = os.path.join(
                output_dir,
                f"{base_name}_mask.png"
            )

            cv2.imwrite(mask_path, mask)

            print(f"Saved binary mask: {mask_path}")

        # --------------------------------------------------
        # CONNECTED COMPONENTS
        # --------------------------------------------------

        labeled_mask = measure.label(mask)

        regions = measure.regionprops(labeled_mask)

        # --------------------------------------------------
        # FILTER + STORE CENTROIDS
        # --------------------------------------------------

        min_area = 1000

        region_data = []

        for region in regions:

            if region.area < min_area:
                continue

            cy, cx = region.centroid

            region_data.append({
                "region": region,
                "cy": cy,
                "cx": cx
            })

        # --------------------------------------------------
        # SORT TOP → BOTTOM
        # --------------------------------------------------

        region_data = sorted(
            region_data,
            key=lambda x: x["cy"]
        )

        # --------------------------------------------------
        # GROUP INTO ROWS
        # --------------------------------------------------

        row_threshold = 80

        rows = []

        for item in region_data:

            placed = False

            for row in rows:

                if abs(
                    item["cy"] - row[0]["cy"]
                ) < row_threshold:

                    row.append(item)

                    placed = True

                    break

            if not placed:
                rows.append([item])

        # --------------------------------------------------
        # SORT LEFT → RIGHT INSIDE ROWS
        # --------------------------------------------------

        sorted_regions = []

        for row in rows:

            row = sorted(
                row,
                key=lambda x: x["cx"]
            )

            sorted_regions.extend(
                [x["region"] for x in row]
            )

        # Replace regions
        regions = sorted_regions

        # --------------------------------------------------
        # FILE METADATA
        # --------------------------------------------------

        file = os.path.basename(hdr_path)

        plot_code = file[20:25]

        lk_id = PLOTNAME_TO_LKID.get(plot_code)

        if lk_id is None:

            print(
                f"Skipping file (plot code not found): {file}"
            )

            return

        plot_number = file[15:25]

        date_code = file[26:34]

        new_name = f"{lk_id}_{plot_number}_{date_code}"

        crop_dir = os.path.join(
            output_dir,
            f"{new_name}"
        )

        os.makedirs(crop_dir, exist_ok=True)

        # --------------------------------------------------
        # CROP OBJECTS
        # --------------------------------------------------

        saved_count = 0

        margin = 10

        for i, region in enumerate(regions):

            minr, minc, maxr, maxc = region.bbox

            # Add margin
            minr = max(minr - margin, 0)

            minc = max(minc - margin, 0)

            maxr = min(
                maxr + margin,
                seg_bgr.shape[0])
            maxc = min(
                maxc + margin,
                seg_bgr.shape[1])
            # --------------------------------------------------
            # HSI CROP
            # --------------------------------------------------

            crop = seg_bgr[
                minr:maxr,
                minc:maxc,
                :].copy()


            crop_path = os.path.join(
                crop_dir,
                f"{new_name}_{date_code}_{saved_count:03d}.tiff"
            )

            tifffile.imwrite(
                crop_path,
                crop.astype(np.float32)
            )

            del crop

            # --------------------------------------------------
            # RGB CROP
            # --------------------------------------------------

            crop_rgb = seg_bgr[
                minr:maxr,
                minc:maxc,
                rgb_rgb
            ].copy()

            crop_rgb = np.clip(
                crop_rgb * 255,
                0,
                255
            ).astype(np.uint8)

            crop_mask = mask[
                minr:maxr,
                minc:maxc
            ]

            crop_rgb[crop_mask == 0] = [255, 255, 255]

            rgb_path = os.path.join(
                crop_dir,
                f"{new_name}_{date_code}_{saved_count:03d}.png"
            )

            plt.imsave(
                rgb_path,
                crop_rgb
            )

            del crop_rgb
            del crop_mask

            print(
                f"Saved crop {saved_count:03d}"
            )

            saved_count += 1

        print(
            f"Saved {saved_count} cropped plants "
            f"to: {crop_dir}"
        )

    except Exception as e:

        print(
            f"Error processing {hdr_path}: {str(e)}"
        )

    finally:

        gc.collect()

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    root_folder = (
        r"D:\Hyperspec\Trials\20210521\FX10_VNIR\New folder"
    )

    output_root = (
        r"D:\Hyperspec\Trials\20210521\FX10_VNIR\New folder\New folder (2)"
    )

    for root, dirs, files in os.walk(root_folder):

        for file in files:

            if file.endswith(".hdr"):

                hdr_path = os.path.join(root, file)

                segment_hsi_by_exg_ndvi(
                    hdr_path,
                    output_root
                )

                gc.collect()

# ==========================================================
# END
# ==========================================================

print("\nDONE")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ========== Segment ENVI cubes using pre‑trained Random Forest ==========
import numpy as np
import joblib
import os
import glob
import spectral.io.envi as envi
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Pseudo‑RGB generation (from your function, slightly adapted)
# ------------------------------------------------------------
def envi_to_pseudo_rgb_png(hdr_path, red_band, green_band, blue_band, output_rgb_png):
    """
    Load ENVI cube and save a pseudo‑RGB PNG.
    (No full TIFF output to keep it simple.)
    """
    img = envi.open(hdr_path)
    cube = img.load()  # shape: (rows, cols, bands)
    print(f'  Cube shape for RGB: {cube.shape}')

    red = cube[:, :, red_band].squeeze()
    green = cube[:, :, green_band].squeeze()
    blue = cube[:, :, blue_band].squeeze()

    def normalise(band):
        band_min = band.min()
        band_max = band.max()
        if band_max - band_min == 0:
            return np.zeros_like(band, dtype=np.uint8)
        return ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)

    rgb = np.stack([normalise(red), normalise(green), normalise(blue)], axis=2)
    Image.fromarray(rgb, mode='RGB').save(output_rgb_png)
    print(f'  Pseudo‑RGB PNG saved to {output_rgb_png}')


# ------------------------------------------------------------
# Segment one ENVI cube using the trained model
# ------------------------------------------------------------
def segment_envi_cube(model, hdr_path):
    """
    Load ENVI cube, classify each pixel, return binary mask.
    """
    img = envi.open(hdr_path)
    cube = img.load()  # shape: (rows, cols, bands)
    print(f'  Loaded cube shape: {cube.shape}')

    h, w, bands = cube.shape
    pixels = cube.reshape(-1, bands)
    print(f'  Classifying {pixels.shape[0]} pixels...')

    pred = model.predict(pixels)
    mask = pred.reshape(h, w).astype(np.uint8)
    return mask


def save_mask(mask, output_path):
    """Save mask as PNG (0=black, 1=white)."""
    img = (mask * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)
    print(f'  Mask saved to {output_path}')


# ------------------------------------------------------------
# Main: loop over all .hdr files in a folder
# ------------------------------------------------------------
def main():
    # --- Configuration (EDIT THESE PATHS) ---
    MODEL_PATH = r"D:\Hyperspec\Trials\20210521\FX10_VNIR\New folder\random_forest_model.pkl"
    CUBES_FOLDER = r"D:\Hyperspec\Trials\20210521\FX10_VNIR\New folder"
    OUTPUT_SUBFOLDER = "segmentation_masks"   # masks + RGBs will be placed here

    # Band indices for pseudo‑RGB (adjust to your data)
    RED_BAND = 150
    GREEN_BAND = 100
    BLUE_BAND = 50

    # Whether to also generate pseudo‑RGB PNGs
    GENERATE_RGB = True

    # --------------------------------------------------------
    # Load pre‑trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded.\n")

    # Create output directory
    output_dir = os.path.join(CUBES_FOLDER, OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Find all .hdr files (case‑insensitive)
    hdr_files = glob.glob(os.path.join(CUBES_FOLDER, "*.hdr")) + \
                glob.glob(os.path.join(CUBES_FOLDER, "*.HDR"))
    hdr_files = sorted(set(hdr_files))

    if not hdr_files:
        print(f"No .hdr files found in {CUBES_FOLDER}")
        return

    print(f"Found {len(hdr_files)} ENVI cube(s).\n")

    # Process each cube
    for hdr_path in hdr_files:
        base_name = os.path.splitext(os.path.basename(hdr_path))[0]
        print(f"\n>>> Processing {base_name}")

        try:
            # 1. Segmentation
            mask = segment_envi_cube(model, hdr_path)
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            save_mask(mask, mask_path)

            # 2. Optional pseudo‑RGB
            if GENERATE_RGB:
                rgb_path = os.path.join(output_dir, f"{base_name}_pseudo_rgb.png")
                envi_to_pseudo_rgb_png(hdr_path, RED_BAND, GREEN_BAND, BLUE_BAND, rgb_path)

            # Optional: display mask (comment out if too many)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Mask: {base_name}")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nAll done. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()

from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd
import os
import json
import cv2
import numpy as np
from skimage import measure
import tifffile
import matplotlib.pyplot as plt
from spectral import open_image
import gc

def cm_analysis(y_true, y_pred, filename, labels, classes, ymap=None, figsize=(5,5)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=2.5)

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            #elif c == 0:
            #    annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'predicted Label'
    cm.columns.name = 'True Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')

    sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap = 'Blues')# cmap="Blues")
    plt.savefig(filename,  bbox_inches='tight')

# import pandas as pd
df = pd.read_excel(r'E:\confusion matrix2.xlsx',sheet_name=0, engine='openpyxl')
# print(df)
y_pred= df.iloc[:, -1].values
y_true = df.iloc[:,0].values
# #y_true = [1, 0, 1, 1, 0, 1,2,2,2,1,1,1,3,3,2,2,2,1,1,3,3,3,2,2,1]
# #y_pred = [0, 0, 1, 1, 0, 1,2,2,2,2,2,1,2,3,3,2,3,1,1,3,3,3,1,2,1]
filename= 'Anthocyanin Levels '
labels=[0,1]
# classes= ['0','1', '2']
# labels=[0,1]
classes= ['SNP 0','SNP 2']
cm_analysis(y_true, y_pred, filename, labels, classes, ymap=None, figsize=(8,6))


# +++++++++++++++++++++++++++++++++++++++++++++===========================================================
import pandas as pd
import os
import json
import cv2
import numpy as np
from skimage import measure, morphology
import tifffile
import matplotlib.pyplot as plt
from spectral import open_image
import gc

# Load full mapping from JSON file
with open('plotname_to_lkid_new.json', 'r') as f:
    PLOTNAME_TO_LKID = json.load(f)

# ----------------------------------------------------------------------
# Helper: wavelength list (adjust to your sensor)
def get_wavelengths():
    # Return a list of 224 wavelengths (replace with actual values)
    # For now, return generic indices
    return list(range(224))

# Helper: plot a mean spectrum
def plot_spectrum(mean_spectrum, wavelengths, title, save_path, ylabel='Reflectance (a.u.)'):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, mean_spectrum, 'b-', linewidth=1.5)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# Helper: save pseudo‑RGB as a plot (with axes, title)
def save_pseudo_rgb_plot(rgb_image, title, save_path, vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 1.0
    plt.imshow(rgb_image, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Helper: plot all plant spectra together (overlay)
def plot_all_spectra(spectra_list, wavelengths, title, save_path, labels=None):
    plt.figure(figsize=(12, 8))
    for i, spectrum in enumerate(spectra_list):
        label = labels[i] if labels else f"Plant {i+1}"
        plt.plot(wavelengths, spectrum, linewidth=1, alpha=0.7, label=label)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance (a.u.)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ----------------------------------------------------------------------
def segment_hsi_by_exg_ndvi(hdr_path, output_dir,
                            save_mask=True, save_ndvi=True,
                            save_segmented_hsi=True, save_spectra_plots=True,
                            save_pseudo_rgb=True):
    """
    Process a single .hdr file:
    - Compute NDVI and ExGR, combine masks, clean, apply to HSI.
    - Save NDVI mask, binary mask, segmented full cube (TIFF), RGB preview.
    - Extract and plot mean spectrum of the whole plot.
    - Extract and plot pseudo‑RGB of original and segmented HSI using bands (85,60,40).
    - Crop individual plants, save as .tiff (HSI) and .png (RGB), and plot their mean spectra.
    - Also plot all individual plant spectra together in one figure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # --- Step 1: Load and trim hyperspectral image ---
        print(f"Loading HSI: {hdr_path}")
        img = open_image(hdr_path)
        hsi = img.load()
        im = hsi[:, :, :224].astype(np.float32)  # first 224 bands
        del img, hsi
        gc.collect()

        # Wavelength list (for spectral plots)
        wavelengths = get_wavelengths()
        if len(wavelengths) != im.shape[2]:
            wavelengths = list(range(im.shape[2]))

        # --- Band indices for different purposes ---
        rgb_bands = (85, 59, 17)      # bands for RGB preview (segmented)
        ndvi_bands = (163, 85, 85)    # (NIR, Red) for NDVI
        rgb_rgb = (85, 60, 40)        # bands for cropped RGB and pseudo‑RGB (better colour balance)
        pseudo_rgb_bands = rgb_rgb    # use same for pseudo‑RGB

        # --- Extract original pseudo‑RGB (before any processing) ---
        base_name = os.path.splitext(os.path.basename(hdr_path))[0]
        if save_pseudo_rgb:
            # Normalise the three bands to [0,1] for display
            def norm_band(b):
                b = b.astype(np.float32)
                return (b - b.min()) / (b.max() - b.min() + 1e-8)
            r_orig = norm_band(im[:, :, pseudo_rgb_bands[0]])
            g_orig = norm_band(im[:, :, pseudo_rgb_bands[1]])
            b_orig = norm_band(im[:, :, pseudo_rgb_bands[2]])
            pseudo_rgb_orig = np.stack([r_orig, g_orig, b_orig], axis=2)
            title_orig = f"Original pseudo‑RGB (bands {pseudo_rgb_bands}) – {base_name}"
            out_pseudo_orig = os.path.join(output_dir, f"{base_name}_original_pseudo_rgb.png")
            save_pseudo_rgb_plot(pseudo_rgb_orig, title_orig, out_pseudo_orig)
            print(f"Saved original pseudo‑RGB plot: {out_pseudo_orig}")

        # --- NDVI and ExGR calculations (same as before) ---
        R = im[:, :, rgb_bands[0]]
        G = im[:, :, rgb_bands[1]]
        B = im[:, :, rgb_bands[2]]
        IR = im[:, :, ndvi_bands[0]]
        Red = im[:, :, ndvi_bands[1]]

        NDVI = (IR - Red) / (IR + Red + 1e-6)
        del IR, Red
        NDVI_norm = cv2.normalize(NDVI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_ndvi = cv2.threshold(NDVI_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if save_ndvi:
            ndvi_vis = cv2.normalize(NDVI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ndvi_path = os.path.join(output_dir, f"{base_name}_NDVI.png")
            cv2.imwrite(ndvi_path, ndvi_vis)
            print(f"Saved NDVI mask: {ndvi_path}")
        del NDVI, NDVI_norm

        ExG = (2 * G) - R - B
        ExGR = ExG - ((1.4 * R) - G)
        del R, G, B, ExG
        ExGR_norm = cv2.normalize(ExGR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_exgr = cv2.threshold(ExGR_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        del ExGR, ExGR_norm

        combined_mask = cv2.bitwise_and(mask_ndvi, mask_exgr)
        del mask_ndvi, mask_exgr

        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = morphology.remove_small_objects(cleaned_mask.astype(bool), min_size=500)
        mask = cleaned_mask.astype(np.uint8) * 255
        del combined_mask, kernel, cleaned_mask

        if save_mask:
            mask_path = os.path.join(output_dir, f"{base_name}_binary_mask.png")
            cv2.imwrite(mask_path, mask)
            print(f"Saved binary mask: {mask_path}")

        # Apply mask to the whole HSI cube
        seg_bgr = im * (mask[:, :, np.newaxis] / 255.0)
        del im
        seg_bgr = cv2.normalize(seg_bgr, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if save_segmented_hsi:
            seg_cube_path = os.path.join(output_dir, f"{base_name}_segmented_cube.tiff")
            tifffile.imwrite(seg_cube_path, seg_bgr.astype(np.float32))
            print(f"Saved segmented HSI cube: {seg_cube_path}")

        # --- Pseudo‑RGB of the segmented HSI (using the same bands) ---
        if save_pseudo_rgb:
            r_seg = seg_bgr[:, :, pseudo_rgb_bands[0]]
            g_seg = seg_bgr[:, :, pseudo_rgb_bands[1]]
            b_seg = seg_bgr[:, :, pseudo_rgb_bands[2]]
            # Already normalised in [0,1] from seg_bgr normalisation
            pseudo_rgb_seg = np.stack([r_seg, g_seg, b_seg], axis=2)
            # Set background (mask=0) to white for better visualisation
            pseudo_rgb_seg[mask == 0] = [1.0, 1.0, 1.0]
            title_seg = f"Segmented pseudo‑RGB – {base_name}"
            out_pseudo_seg = os.path.join(output_dir, f"{base_name}_segmented_pseudo_rgb.png")
            save_pseudo_rgb_plot(pseudo_rgb_seg, title_seg, out_pseudo_seg)
            print(f"Saved segmented pseudo‑RGB plot: {out_pseudo_seg}")

        # --- RGB preview of segmented image (using rgb_bands) – also saved as plot
        masked_rgb = seg_bgr[:, :, rgb_bands]
        masked_rgb = np.clip(masked_rgb * 255, 0, 255).astype(np.uint8)
        masked_rgb[mask == 0] = [255, 255, 255]
        rgb_preview_path = os.path.join(output_dir, f"{base_name}_segmented_rgb.png")
        plt.imsave(rgb_preview_path, masked_rgb)
        print(f"Saved segmented RGB preview: {rgb_preview_path}")

        # --- Mean spectrum of whole plot ---
        if save_spectra_plots:
            plant_pixels = seg_bgr[mask > 0, :]
            if plant_pixels.shape[0] > 0:
                mean_spectrum = np.mean(plant_pixels, axis=0)
                title = f"Mean spectrum - {base_name}"
                spec_path = os.path.join(output_dir, f"{base_name}_mean_spectrum.png")
                plot_spectrum(mean_spectrum, wavelengths, title, spec_path)
            else:
                print(f"No plant pixels found for {base_name} – skipping spectrum plot.")

        # --- Crop individual plants and save ---
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        file = os.path.basename(hdr_path)
        plot_code = file[20:25]   # adjust to your filename pattern
        lk_id = PLOTNAME_TO_LKID.get(plot_code)
        if lk_id is None:
            print(f" Skipping file (plot code not found): {file}")
            return

        plot_number = file[15:25]   # adjust
        date_code = file[26:34]     # adjust
        new_name = f"{lk_id}_{plot_number}_{date_code}"
        crop_dir = os.path.join(output_dir, f"{new_name}")
        os.makedirs(crop_dir, exist_ok=True)

        min_area = 1000
        saved_count = 0
        margin = 10
        plant_spectra_list = []   # collect mean spectra for overlay plot
        plant_labels_list = []    # collect labels

        for i, region in enumerate(regions):
            if region.area < min_area:
                continue

            minr, minc, maxr, maxc = region.bbox
            minr = max(minr - margin, 0)
            minc = max(minc - margin, 0)
            maxr = min(maxr + margin, seg_bgr.shape[0])
            maxc = min(maxc + margin, seg_bgr.shape[1])

            # Crop HSI
            crop = seg_bgr[minr:maxr, minc:maxc, :].copy()
            crop_path = os.path.join(crop_dir, f"{new_name}_{date_code}_{saved_count:03d}.tiff")
            tifffile.imwrite(crop_path, crop.astype(np.float32))

            # Crop RGB preview
            crop_rgb = seg_bgr[minr:maxr, minc:maxc, rgb_rgb].copy()
            crop_rgb = np.clip(crop_rgb * 255, 0, 255).astype(np.uint8)
            crop_mask = mask[minr:maxr, minc:maxc]
            crop_rgb[crop_mask == 0] = [255, 255, 255]
            rgb_path = os.path.join(crop_dir, f"{new_name}_{date_code}_{saved_count:03d}.png")
            plt.imsave(rgb_path, crop_rgb)

            # Plot mean spectrum of this individual plant
            if save_spectra_plots:
                plant_pixels_crop = crop[crop_mask > 0, :]
                if plant_pixels_crop.shape[0] > 0:
                    mean_spectrum_crop = np.mean(plant_pixels_crop, axis=0)
                    plant_spectra_list.append(mean_spectrum_crop)
                    plant_labels_list.append(f"{new_name}_{date_code}_{saved_count:03d}")
                    title_crop = f"Mean spectrum - {new_name}_{date_code}_{saved_count:03d}"
                    spec_crop_path = os.path.join(crop_dir, f"{new_name}_{date_code}_{saved_count:03d}_spectrum.png")
                    plot_spectrum(mean_spectrum_crop, wavelengths, title_crop, spec_crop_path)

            del crop, crop_rgb, crop_mask
            saved_count += 1

        # --- Plot all plant spectra together (overlay) ---
        if save_spectra_plots and len(plant_spectra_list) > 0:
            all_spec_path = os.path.join(crop_dir, f"{new_name}_all_plants_spectra.png")
            plot_all_spectra(plant_spectra_list, wavelengths,
                             f"Mean spectra of all plants – {new_name}",
                             all_spec_path, labels=plant_labels_list)
            print(f"Saved overlay of all plant spectra: {all_spec_path}")

        print(f"Saved {saved_count} cropped plants with margin to: {crop_dir}")

    except Exception as e:
        print(f"Error processing {hdr_path}: {str(e)}")
    finally:
        gc.collect()


if __name__ == "__main__":
    # --- Set your input and output folders ---
    root_folder = r"E:\Utrecht U-files\New folder (2)\New folder"
    output_root = r"E:\Utrecht U-files\New folder (2)\New folder\New folder"

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".hdr"):
                hdr_path = os.path.join(root, file)
                segment_hsi_by_exg_ndvi(hdr_path, output_root,
                                        save_mask=True,
                                        save_ndvi=True,
                                        save_segmented_hsi=True,
                                        save_spectra_plots=True,
                                        save_pseudo_rgb=True)
                gc.collect()
# ____________________________________________________________________________________________________________

print()
# SEGMENTATION OF HSI DATA USING TWO DIFFERENT SPECTRAL POINTS
# _________________________________________________________________________________#
# Load full mapping from JSON file
with open('plotname_to_lkid_new.json', 'r') as f:
    PLOTNAME_TO_LKID = json.load(f)


def segment_hsi_by_exg_ndvi(hdr_path, output_dir, save_mask=False):
    try:
        os.makedirs(output_dir, exist_ok=True)

        # --- Step 1: Load and trim hyperspectral image ---
        print(f"Loading HSI: {hdr_path}")

        img = open_image(hdr_path)
        hsi = img.load()
        im = hsi[:, :, :224].astype(np.float32)

        del img, hsi
        gc.collect()

        # Band indices
        rgb_bands = (85, 59, 17)
        ndvi_bands = (163, 85, 85)
        rgb_rgb = (150, 100, 57)

        R = im[:, :, rgb_bands[0]]
        G = im[:, :, rgb_bands[1]]
        B = im[:, :, rgb_bands[2]]
        IR = im[:, :, ndvi_bands[0]]
        Red = im[:, :, ndvi_bands[1]]

        NDVI = (IR - Red) / (IR + Red + 1e-6)
        del IR, Red
        NDVI_norm = cv2.normalize(NDVI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_ndvi = cv2.threshold(NDVI_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        del NDVI, NDVI_norm

        ExG = (2 * G) - R - B
        ExGR = ExG - ((1.4 * R) - G)
        del R, G, B, ExG
        ExGR_norm = cv2.normalize(ExGR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_exgr = cv2.threshold(ExGR_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        del ExGR, ExGR_norm

        combined_mask = cv2.bitwise_and(mask_ndvi, mask_exgr)
        del mask_ndvi, mask_exgr

        # Morphological cleaning
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = morphology.remove_small_objects(cleaned_mask.astype(bool), min_size=500)
        mask = cleaned_mask.astype(np.uint8) * 255

        del combined_mask, kernel, cleaned_mask

        # Apply mask
        seg_bgr = im * (mask[:, :, np.newaxis] / 255.0)
        del im
        seg_bgr = cv2.normalize(seg_bgr, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        masked_rgb = seg_bgr[:, :, rgb_bands]
        masked_rgb = np.clip(masked_rgb * 255, 0, 255).astype(np.uint8)
        masked_rgb[mask == 0] = [255, 255, 255]

        base_name = os.path.splitext(os.path.basename(hdr_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_segmented_rgb.png")
        plt.imsave(out_path, masked_rgb)
        print(f"Saved segmented RGB: {out_path}")
        del masked_rgb

        if save_mask:
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask)
            print(f"Saved binary mask: {mask_path}")

        # Crop and save
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        file = os.path.basename(hdr_path)
        plot_code = file[20:25]
        lk_id = PLOTNAME_TO_LKID.get(plot_code)
        if lk_id is None:
            print(f" Skipping file (plot code not found): {file}")
            return

        plot_number = file[15:25]
        date_code = file[26:34]
        new_name = f"{lk_id}_{plot_number}"
        crop_dir = os.path.join(output_dir, f"{new_name}")
        os.makedirs(crop_dir, exist_ok=True)

        min_area = 1000
        saved_count = 0
        margin = 10

        for i, region in enumerate(regions):
            if region.area < min_area:
                continue

            minr, minc, maxr, maxc = region.bbox
            minr = max(minr - margin, 0)
            minc = max(minc - margin, 0)
            maxr = min(maxr + margin, seg_bgr.shape[0])
            maxc = min(maxc + margin, seg_bgr.shape[1])

            crop = seg_bgr[minr:maxr, minc:maxc, :].copy()
            crop_path = os.path.join(crop_dir, f"{new_name}_{date_code}_{saved_count:03d}.tiff")
            print(crop_path)
            tifffile.imwrite(crop_path, crop.astype(np.float32))
            del crop

            crop_rgb = seg_bgr[minr:maxr, minc:maxc, rgb_rgb].copy()
            crop_rgb = np.clip(crop_rgb * 255, 0, 255).astype(np.uint8)
            crop_mask = mask[minr:maxr, minc:maxc]
            crop_rgb[crop_mask == 0] = [255, 255, 255]

            rgb_path = os.path.join(crop_dir, f"{new_name}_{date_code}_{saved_count:03d}.png")
            plt.imsave(rgb_path, crop_rgb)
            del crop_rgb, crop_mask

            saved_count += 1

        print(f"Saved {saved_count} cropped plants with margin to: {crop_dir}")

    except Exception as e:
        print(f"Error processing {hdr_path}: {str(e)}")
    finally:
        gc.collect()


if __name__ == "__main__":
    root_folder = r"E:\Utrecht U-files\New folder (2)\New folder"
    output_root = r"E:\Utrecht U-files\New folder (2)\New folder\New folder"

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".hdr"):
                hdr_path = os.path.join(root, file)
                segment_hsi_by_exg_ndvi(hdr_path, output_root)
                gc.collect()

# ____________________________END OF SCRIPT_____________________________________
print()
from matplotlib import pyplot as plt
import spectral
import pandas as pd
import numpy as np
from skimage import io
import tifffile
import spectral.io.envi as envi
import os
import imageio
import cv2
print()
from matplotlib import pyplot as plt
import pandas as pd

# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Assuming the file is named 'data.xlsx' and is in the current directory
df = pd.read_excel('D:/classification_spectral.xlsx')

# split into input and output columns
X = df.iloc[:, 0:224]
y = df.iloc[:, -1]
X = np.asarray(X)
y = np.asarray(y)

# Print the first 5 rows of the DataFrame
print(df.head())
# prepare configuration for cross validation test harness
seed = 7
shuffle = True
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=101, shuffle= True)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
ax.set_xlabel("Models")
ax.set_ylabel("Accuracy")
legend = ax.legend(loc='upper right')
plt.show()


def cropping_HSIData(path):
    global Data_H, Data, filename
    for root, subdirs, files in os.walk(path, topdown=False):
        # calling the files
        n = 0
        a=[]
        for filename in files:
            # some filename starts with ('.') hence this much be first corrected before calling the giving the filename
            # reading from json file and extracting some information
            if filename.endswith('.tiff') or filename.endswith('.tif'):
                Data = os.path.join(root, filename)
                img = io.imread(Data)
                #plt.imshow(img[:,:,(150,50,10)])
                plt.imshow(img[:, :, (100, 30, 10)])
                plt.show()
                # Assuming 'normalized_img_array' is your hyperspectral data cube
                # and you want to plot the spectrum for the pixel at coordinates (x, y)
    #             x = 41
    #             y = 133
    #
    #             v = img[x, y, :]
    #             img_array = v
    #             min_val = img_array.min()
    #             max_val = img_array.max()
    #
    #             # Normalize the data
    #             a = (img_array - min_val) / (max_val - min_val)
    #
    #             x2 = 112
    #             y2 = 85
    #             v2 = img[x2, y2, :]
    #             img_array2 = v2
    #             min_val2 = img_array2.min()
    #             max_val2 = img_array2.max()
    #             a2 = (img_array2 - min_val2) / (max_val2 - min_val2)
    #
    #             x3 = 199
    #             y3 = 108
    #             v3 = img[x3, y3, :]
    #             img_array3 = v3
    #             min_val3 = img_array3.min()
    #             max_val3 = img_array3.max()
    #             a3 = (img_array3 - min_val3) / (max_val3 - min_val3)
    #
    #             x4 = 290
    #             y4 = 135
    #             v4 = img[x4, y4, :]
    #             img_array4 = v4
    #             min_val4 = img_array4.min()
    #             max_val4 = img_array4.max()
    #             a4 = (img_array4 - min_val4) / (max_val4 - min_val4)
    #
    #             x5 = 43
    #             y5 = 326
    #             v5 = img[x5, y5, :]
    #             img_array5 = v5
    #             min_val5 = img_array5.min()
    #             max_val5 = img_array5.max()
    #             a5 = (img_array5 - min_val5) / (max_val5 - min_val5)
    #
    #             x6 = 199
    #             y6 = 327
    #             v6 = img[x6, y6, :]
    #             img_array6 = v6
    #             min_val6 = img_array6.min()
    #             max_val6 = img_array6.max()
    #             a6 = (img_array6 - min_val6) / (max_val6 - min_val6)
    #
    #             # plt.plot(a)
    #             # plt.plot(a2)
    #             # plt.plot(a3)

    #             p= np.stack((a,a2,a3,a4,a5,a6))
    #
    # # Assuming 'v' is the spectral signature you want to save
    # df = pd.DataFrame(p)
    # df=df.T
    #
    # # Save the DataFrame to an Excel file
    # df.to_excel(r'D:\Utrecht U-files' + filename[:-4]+ '.xlsx', header= False, index=False)

path = r'D:\Utrecht U-files\Sativa'
cropping_HSIData(path)


