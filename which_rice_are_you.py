"""
Which rice are you?
Loads any image (yes, including a selfie) and tells you which rice variety
the model thinks you are.

Usage:
    python which_rice_are_you.py <image_path> [--model <model_path>]

Examples:
    python which_rice_are_you.py me.jpg
    python which_rice_are_you.py me.jpg --model models/rice_vit.keras
"""

import argparse
import sys

import numpy as np

CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

DESCRIPTIONS = {
    "Arborio":   "Short, plump, and creamy — you bring comfort to every situation. 🍚",
    "Basmati":   "Long, slender, and fragrant — you stand out in any crowd. 🌾",
    "Ipsala":    "Versatile and reliable — everyone can count on you. 💪",
    "Jasmine":   "Soft, aromatic, and subtly sweet — you have a magnetic personality. 🌸",
    "Karacadag": "Rare, distinctive, and full of character — you're one of a kind. ✨",
}


def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load an image, resize to 224x224, and normalise to [0, 1]."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is required: pip install Pillow")

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def predict(model_path: str, image: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Run inference and return (class_name, confidence, all_probs)."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except ImportError:
        sys.exit("TensorFlow is required: pip install tensorflow")

    model = keras.models.load_model(model_path)
    probs = model.predict(image, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


def main():
    parser = argparse.ArgumentParser(description="Which rice are you?")
    parser.add_argument("image", help="Path to the input image (jpg/png/…)")
    parser.add_argument(
        "--model",
        default="models/rice_vit.keras",
        help="Path to the trained Keras model file (default: models/rice_vit.keras)",
    )
    args = parser.parse_args()

    print(f"\nLoading model from: {args.model}")
    print(f"Analysing image:    {args.image}\n")

    image = load_and_preprocess(args.image)
    rice, confidence, probs = predict(args.model, image)

    print("=" * 50)
    print(f"  You are... {rice} rice!")
    print(f"  Confidence: {confidence * 100:.1f}%")
    print()
    print(f"  {DESCRIPTIONS[rice]}")
    print("=" * 50)

    print("\nFull probability breakdown:")
    for name, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:<12} {prob * 100:5.1f}%  {bar}")

    print()


if __name__ == "__main__":
    main()
