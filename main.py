"""
Main entry point for the HSI 2D CNN Classifier.
Loads trained model and runs prediction API or test prediction.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.inference_2d import HSI2DClassifier

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")


def test_prediction(model_path: str, sample_image_path: str = None):
    """Test the model with a sample image."""
    print("=" * 60)
    print("TESTING MODEL PREDICTION")
    print("=" * 60)
    
    print(f"Loading model from: {model_path}")
    classifier = HSI2DClassifier(model_path)
    print(f"Model loaded. Input shape: {classifier.input_shape}")
    print(f"Classes: {classifier.class_names}")
    
    if sample_image_path and os.path.exists(sample_image_path):
        print(f"Loading test image from: {sample_image_path}")
        if sample_image_path.endswith('.npy'):
            image = np.load(sample_image_path).astype(np.float32)
        else:
            import tifffile as tiff
            image = tiff.imread(sample_image_path).astype(np.float32)
        if image.shape[0] == 155 and len(image.shape) == 3:
            image = np.moveaxis(image, 0, -1)
    else:
        print("Creating synthetic test image...")
        image = np.random.randn(15, 15, 155).astype(np.float32)
    
    result = classifier.predict(image)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Prediction: {result['prediction']}")
    print(f"Class ID: {result['class_id']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"Processing time: {result['processing_time_ms']:.2f} ms")
    print("\nAll probabilities:")
    for cls, prob in result['all_probabilities'].items():
        print(f"  {cls}: {prob:.4f} ({prob*100:.2f}%)")
    print("=" * 60)
    
    return result


def run_api(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    print("=" * 60)
    print("STARTING API SERVER")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"API Docs: http://localhost:{port}/docs")
    print("=" * 60)
    
    import uvicorn
    from src.api.routes_2d import create_app
    
    app = create_app(model_path)
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="HSI 2D CNN Classifier - Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test the model
  python main.py --mode test
  
  # Run the API server
  python main.py --mode api --port 8000
        """
    )
    
    parser.add_argument("--model_path", type=str, default="./models/model.keras")
    parser.add_argument("--mode", type=str, default="api", choices=["test", "api"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        sys.exit(1)
    
    if args.mode == "test":
        test_prediction(args.model_path, args.image)
    else:
        run_api(args.model_path, args.host, args.port)


if __name__ == "__main__":
    main()