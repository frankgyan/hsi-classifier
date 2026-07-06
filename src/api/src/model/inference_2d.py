"""
Production inference wrapper for 2D CNN HSI classifier.
"""

import tensorflow as tf
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

class HSI2DClassifier:
    """
    Production-ready classifier for hyperspectral images.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        input_shape: tuple = (15, 15, 155)
    ):
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.class_names = class_names or ['HNHP', 'HNLP', 'LNHP', 'LNLP']
        self.num_classes = len(self.class_names)
        
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded. Input shape: {self.model.input_shape}")
        
        self._warmup()
        print("Model warmed up.")
    
    def _warmup(self):
        dummy_input = np.random.randn(1, *self.input_shape).astype(np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.shape != self.input_shape:
            raise ValueError(f"Expected {self.input_shape}, got {image.shape}")
        
        mean = np.mean(image, axis=2, keepdims=True)
        std = np.std(image, axis=2, keepdims=True) + 1e-8
        return (image - mean) / std
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()
        
        if image.shape != self.input_shape:
            raise ValueError(f"Expected {self.input_shape}, got {image.shape}")
        
        normalized = self.preprocess(image)
        input_batch = np.expand_dims(normalized, axis=0)
        
        probabilities = self.model.predict(input_batch, verbose=0)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = np.max(probabilities, axis=1)[0]
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'status': 'success',
            'prediction': self.class_names[predicted_class],
            'class_id': int(predicted_class),
            'confidence': float(confidence),
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(self.class_names, probabilities[0])
            },
            'processing_time_ms': round(processing_time, 2)
        }
    
    def predict_batch(self, images: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()
        
        if images.shape[1:] != self.input_shape:
            raise ValueError(f"Expected {self.input_shape}, got {images.shape[1:]}")
        
        normalized = np.array([self.preprocess(img) for img in images])
        probabilities = self.model.predict(normalized, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        results = []
        for i in range(len(images)):
            results.append({
                'status': 'success',
                'prediction': self.class_names[predicted_classes[i]],
                'class_id': int(predicted_classes[i]),
                'confidence': float(confidences[i]),
                'all_probabilities': {
                    name: float(prob) 
                    for name, prob in zip(self.class_names, probabilities[i])
                }
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'status': 'success',
            'predictions': results,
            'total_time_ms': round(total_time, 2),
            'num_predictions': len(images)
        }
    
    def predict_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        file_path = Path(file_path)
        
        if file_path.suffix == '.npy':
            image = np.load(file_path).astype(np.float32)
        else:
            import tifffile as tiff
            image = tiff.imread(file_path).astype(np.float32)
        
        if image.shape[0] == 155 and len(image.shape) == 3:
            image = np.moveaxis(image, 0, -1)
        elif image.shape == (15, 15, 155):
            pass
        else:
            raise ValueError(f"Unexpected shape: {image.shape}")
        
        return self.predict(image)