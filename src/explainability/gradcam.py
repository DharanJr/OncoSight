"""
Grad-CAM visual explainability for the CT scan classifier (Module 3).

Shows WHICH REGION of a CT scan the model focused on to make its
prediction — turns "the model says Malignant" into "the model says
Malignant because of this specific region," which is what makes a
prediction clinically trustworthy rather than a black box.

Usage:
    python -m src.explainability.gradcam
        Runs on a few sample images per class from the test split, saves
        individual heatmap panels to outputs/heatmaps/.

    python -m src.explainability.gradcam --image path/to/scan.jpg
        Runs on one specific image.

    python -m src.explainability.gradcam --n-per-class 3
        Change how many test images per class to visualize.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save-only, no GUI window — see evaluate.py for why
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CHECKPOINT_DIR,
    PROCESSED_DATA_DIR,
    HEATMAP_DIR,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from src.models.architectures import build_model, get_target_layer
from src.training.train import get_device

# Grad-CAM explains the best-performing Module 1 model (per project decision)
GRADCAM_MODEL_NAME = "resnet50"


def load_trained_model(model_name: str = GRADCAM_MODEL_NAME):
    device = get_device()
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Train Module 1 first: python -m src.training.train --model resnet50"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_to_idx = checkpoint.get("class_to_idx")
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else None
    return model, device, idx_to_class


def preprocess_image(image_path: Path):
    """
    Returns (normalized_tensor_for_the_model, float_rgb_array_for_display).
    Kept separate on purpose: the model needs ImageNet-normalized input, but
    the heatmap overlay needs a plain 0-1 RGB image to draw on top of.
    """
    image = Image.open(image_path).convert("RGB")
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    image_resized = resize(image)
    rgb_float = np.array(image_resized).astype(np.float32) / 255.0

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    input_tensor = normalize_transform(image_resized).unsqueeze(0)
    return input_tensor, rgb_float


def generate_gradcam_for_image(model, device, target_layer, image_path, save_path, idx_to_class, true_class=None):
    input_tensor, rgb_float = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

    pred_class = idx_to_class[pred_idx] if idx_to_class else str(pred_idx)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_float)
    axes[0].set_title("Original CT scan")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    title = f"Predicted: {pred_class} ({confidence:.1%})"
    if true_class:
        correct = "correct" if pred_class == true_class else "INCORRECT"
        title += f"\nActual: {true_class} ({correct})"
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return pred_class, confidence


def run_on_sample_images(n_per_class: int = 2):
    model, device, idx_to_class = load_trained_model()
    target_layer = get_target_layer(model, GRADCAM_MODEL_NAME)

    test_dir = PROCESSED_DATA_DIR / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"{test_dir} not found. Run `python -m src.data.prepare_dataset` first.")

    print(f"Generating Grad-CAM visualizations using {GRADCAM_MODEL_NAME}...")
    results = []
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        true_class = class_dir.name
        image_paths = sorted(class_dir.glob("*.jpg"))[:n_per_class]
        for image_path in image_paths:
            save_path = HEATMAP_DIR / f"gradcam_{true_class}_{image_path.stem}.png"
            pred_class, confidence = generate_gradcam_for_image(
                model, device, target_layer, image_path, save_path, idx_to_class, true_class
            )
            tag = "OK" if pred_class == true_class else "WRONG"
            print(f"  [{tag}] {image_path.name}: actual={true_class} predicted={pred_class} ({confidence:.1%})")
            results.append((true_class, pred_class, confidence, save_path))

    print(f"\nSaved {len(results)} Grad-CAM visualizations to: {HEATMAP_DIR}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for CT scan predictions")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a single image; if omitted, runs on sample test images per class")
    parser.add_argument("--n-per-class", type=int, default=2)
    args = parser.parse_args()

    if args.image:
        model, device, idx_to_class = load_trained_model()
        target_layer = get_target_layer(model, GRADCAM_MODEL_NAME)
        image_path = Path(args.image)
        save_path = HEATMAP_DIR / f"gradcam_{image_path.stem}.png"
        pred_class, confidence = generate_gradcam_for_image(
            model, device, target_layer, image_path, save_path, idx_to_class
        )
        print(f"Predicted: {pred_class} ({confidence:.1%})")
        print(f"Saved to: {save_path}")
    else:
        run_on_sample_images(n_per_class=args.n_per_class)


if __name__ == "__main__":
    main()