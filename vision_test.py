"""
4iLABS ML Module Recruitment Task
"""
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import urllib.request
import time
import os
import json
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Setup — class labels
# ──────────────────────────────────────────────
if not os.path.exists("imagenet_classes.txt"):
    print("Downloading class labels...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt"
    )
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# ──────────────────────────────────────────────
# MODEL SETUP — ResNet18
# ──────────────────────────────────────────────
print("Loading pre-trained ResNet18 model...")
model = models.resnet18(weights=None)
model.load_state_dict(torch.load("resnet18_weights.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    # TODO 4: Timing logic — measure inference latency in milliseconds
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000.0

    probs = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_idx = torch.max(probs, 0)

    return categories[predicted_idx.item()], confidence.item(), latency_ms


# ──────────────────────────────────────────────
# TODO 1: Turbidity / Haze
# ──────────────────────────────────────────────
def simulate_turbidity(img_array):
    """
    Simulate murky water (blur/haze).
    Input: OpenCV image array (BGR)
    Output: Modified OpenCV image array

    Method:
    - Apply large Gaussian blur (31x31 kernel) to scatter fine details,
      mimicking suspended particles in murky water.
    - Blend with a greenish-white haze layer to simulate silt/phytoplankton
      scatter — real turbid ocean water has a green-white tint.
    """
    # Heavy blur — destroys edges and textures like murky water does
    blurred = cv2.GaussianBlur(img_array, (31, 31), sigmaX=10)

    # Greenish haze layer (murky water tint)
    haze = np.full_like(img_array, fill_value=[180, 200, 160], dtype=np.uint8)

    # Blend: 55% haze, 45% blurred image
    hazy = cv2.addWeighted(blurred, 0.45, haze, 0.55, 0)
    return hazy


# ──────────────────────────────────────────────
# TODO 2: Depth Color Shift
# ──────────────────────────────────────────────
def simulate_color_shift(img_array):
    """
    Simulate depth color loss (attenuate the red channel).
    Input: OpenCV image array (BGR)
    Output: Modified OpenCV image array

    Physics: Water absorbs red light (~620nm) first. By ~10m depth,
    almost no red remains. OpenCV is BGR so red = channel index 2.
    """
    img = img_array.copy().astype(np.float32)

    # Attenuate red channel by 88%
    img[:, :, 2] *= 0.12

    # Slight blue boost (water looks colder/deeper blue-green)
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.2, 0, 255)

    # Slight green boost (algae tint)
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.05, 0, 255)

    return np.clip(img, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# TODO 3: Sensor Noise
# ──────────────────────────────────────────────
def simulate_sensor_noise(img_array):
    """
    Simulate low-light digital camera noise.
    Input: OpenCV image array (BGR)
    Output: Modified OpenCV image array

    Method: Inject zero-mean Gaussian noise into the image array.
    At depth, low ambient light forces the sensor amplifier to boost
    the signal, amplifying thermal and shot noise alongside it.
    std_dev=45 represents a heavily degraded low-light sensor.
    """
    noise = np.random.normal(loc=0.0, scale=45.0, size=img_array.shape)
    noisy = img_array.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# BONUS: MobileNetV2 predict
# ──────────────────────────────────────────────
print("Loading MobileNetV2 (bonus)...")
mob_model = models.mobilenet_v2(weights=None)
mob_model.load_state_dict(torch.load("mobilenetv2_weights.pth", map_location="cpu"))
mob_model.eval()

def predict_mobilenet(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    start_time = time.perf_counter()
    with torch.no_grad():
        output = mob_model(input_tensor)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000.0
    probs = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_idx = torch.max(probs, 0)
    return categories[predicted_idx.item()], confidence.item(), latency_ms


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs("output_images", exist_ok=True)

    # All provided test images
    test_images = [
        "aquarium.jpg",
        "jellyfish.jpg",
        "set_u106_SESR.png",
        "set_f46_SESR.png",
        "set_o20_SESR.png",
        "set_u113_SESR.png",
        "set_f20_SESR.png",
    ]

    all_results = {}

    for base_image in test_images:
        img_path = f"test_images/{base_image}"
        if not os.path.exists(img_path):
            print(f"[SKIP] {base_image} not found")
            continue

        img = cv2.imread(img_path)

        # Write degraded images (named as boilerplate specifies)
        stem = os.path.splitext(base_image)[0]
        turbid_path     = f"output_images/{stem}_turbid.jpg"
        colorshift_path = f"output_images/{stem}_colorshift.jpg"
        noise_path      = f"output_images/{stem}_noise.jpg"

        cv2.imwrite(turbid_path,     simulate_turbidity(img.copy()))
        cv2.imwrite(colorshift_path, simulate_color_shift(img.copy()))
        cv2.imwrite(noise_path,      simulate_sensor_noise(img.copy()))

        images_to_test = [
            ("Baseline (Clean)", img_path),
            ("Turbidity",        turbid_path),
            ("Color Shift",      colorshift_path),
            ("Sensor Noise",     noise_path),
        ]

        print(f"\n{'='*50}")
        print(f"  IMAGE: {base_image}")
        print(f"{'='*50}")
        print("\n--- ResNet18 ---")
        print(f"{'Condition':<20} {'Prediction':<28} {'Confidence':>10}  {'Latency':>10}")
        print("-" * 72)

        img_results = {}
        for condition_name, file_path in images_to_test:
            label, conf, latency = predict(file_path)
            mob_label, mob_conf, mob_latency = predict_mobilenet(file_path)

            img_results[condition_name] = {
                "resnet18":   {"label": label,     "confidence": round(conf*100,2),     "latency_ms": round(latency,2)},
                "mobilenetv2":{"label": mob_label, "confidence": round(mob_conf*100,2), "latency_ms": round(mob_latency,2)},
            }

            print(f"{condition_name:<20} {label:<28} {conf*100:>9.2f}%  {latency:>8.2f}ms")

        print("\n--- MobileNetV2 (Bonus) ---")
        print(f"{'Condition':<20} {'Prediction':<28} {'Confidence':>10}  {'Latency':>10}")
        print("-" * 72)
        for condition_name in img_results:
            d = img_results[condition_name]["mobilenetv2"]
            print(f"{condition_name:<20} {d['label']:<28} {d['confidence']:>9.2f}%  {d['latency_ms']:>8.2f}ms")

        all_results[base_image] = img_results

    # Save full results JSON
    with open("output_images/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[SAVED] output_images/results.json")

    # ── Visualizations ──────────────────────────────
    # 1. Degradation panel for aquarium.jpg
    base = "aquarium.jpg"
    img  = cv2.imread(f"test_images/{base}")
    stem = "aquarium"
    panels = {
        "Clean":       img,
        "Turbidity":   simulate_turbidity(img.copy()),
        "Color Shift": simulate_color_shift(img.copy()),
        "Sensor Noise":simulate_sensor_noise(img.copy()),
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1b2a")
    border_cols = ["#4caf50","#ff9800","#2196f3","#e91e63"]
    for ax, (name, panel_img), bc in zip(axes, panels.items(), border_cols):
        ax.imshow(cv2.cvtColor(panel_img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(name, color="white", fontsize=11, fontweight="bold", pad=6)
        # confidence from results
        cond_key = "Baseline (Clean)" if name == "Clean" else name.replace(" ","_") if " " in name else name
        # map display name back to results key
        key_map = {"Clean":"Baseline (Clean)","Turbidity":"Turbidity",
                   "Color Shift":"Color Shift","Sensor Noise":"Sensor Noise"}
        r = all_results[base][key_map[name]]["resnet18"]
        conf_c = "#4caf50" if r["confidence"]>60 else "#ff9800" if r["confidence"]>30 else "#f44336"
        ax.text(0.5,-0.06, f"{r['label'][:22]}", transform=ax.transAxes,
                ha="center", color="white", fontsize=8)
        ax.text(0.5,-0.13, f"{r['confidence']:.1f}% | {r['latency_ms']:.1f}ms",
                transform=ax.transAxes, ha="center", color=conf_c, fontsize=8.5, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(bc); sp.set_linewidth(3)
    plt.suptitle("aquarium.jpg — Degradation Panel (ResNet18)", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output_images/degradation_panel_aquarium.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1b2a")
    plt.close()

    # 2. Confidence comparison chart across all images — ResNet18
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1b2a")
    conditions = ["Baseline (Clean)", "Turbidity", "Color Shift", "Sensor Noise"]
    cond_short  = ["Clean", "Turbid", "Color\nShift", "Noise"]
    palette     = {"ResNet18":"#4fc3f7", "MobileNetV2":"#f06292"}
    img_labels  = list(all_results.keys())

    for ax_idx, (model_key, model_name) in enumerate(
            [("resnet18","ResNet18"),("mobilenetv2","MobileNetV2")]):
        ax = axes[ax_idx]
        ax.set_facecolor("#111e2d")
        x = np.arange(len(conditions))
        width = 0.8 / len(img_labels)

        for i, img_label in enumerate(img_labels):
            confs = [all_results[img_label][c][model_key]["confidence"] for c in conditions]
            offset = (i - len(img_labels)/2 + 0.5) * width
            bars = ax.bar(x + offset, confs, width,
                          label=img_label, alpha=0.85, edgecolor="white", linewidth=0.3)

        ax.axhline(50, color="#ff6b6b", linestyle="--", lw=1.5, label="50% reliability threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_short, color="white", fontsize=9)
        ax.set_ylabel("Top-1 Confidence (%)", color="#aaaaaa")
        ax.set_ylim(0, 110)
        ax.set_title(model_name, color="white", fontweight="bold", fontsize=11)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#111e2d", edgecolor="#334455", labelcolor="white",
                  fontsize=6.5, loc="upper right")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334455")

    plt.suptitle("Confidence Under Degradation — All Test Images",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output_images/confidence_chart.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1b2a")
    plt.close()

    # 3. Latency comparison — ResNet18 vs MobileNetV2
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#111e2d")
    model_names = ["ResNet18", "MobileNetV2"]
    model_keys  = ["resnet18", "mobilenetv2"]
    cols        = ["#4fc3f7", "#f06292"]
    # Average latency per model per condition across all images
    x = np.arange(len(conditions))
    width = 0.35
    for i, (mkey, mname, col) in enumerate(zip(model_keys, model_names, cols)):
        avg_lats = []
        for c in conditions:
            lats = [all_results[img][c][mkey]["latency_ms"] for img in img_labels]
            avg_lats.append(np.mean(lats))
        bars = ax.bar(x + (i-0.5)*width, avg_lats, width,
                      label=mname, color=col, alpha=0.9, edgecolor="white", linewidth=0.4)
        for bar, l in zip(bars, avg_lats):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{l:.1f}ms", ha="center", va="bottom", color="white", fontsize=8)
    ax.axhline(33.3, color="#ffeb3b", linestyle="--", lw=1.5, label="30 FPS budget (33.3ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_short, color="white", fontsize=10)
    ax.set_ylabel("Avg Inference Latency (ms)", color="#aaaaaa")
    ax.set_title("Latency: ResNet18 vs MobileNetV2 (averaged across all images)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#111e2d", edgecolor="#334455", labelcolor="white", fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334455")
    plt.tight_layout()
    plt.savefig("output_images/latency_chart.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1b2a")
    plt.close()

    print("[SAVED] output_images/degradation_panel_aquarium.png")
    print("[SAVED] output_images/confidence_chart.png")
    print("[SAVED] output_images/latency_chart.png")
    print("\n[DONE] All outputs in ./output_images/")
