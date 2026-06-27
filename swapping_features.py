import argparse
import base64
import json
import re
import sys
import io
from pathlib import Path

import numpy as np
import cv2
import torch
import requests
import ollama
from PIL import Image


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap a furniture component from one image onto another.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # ── Required inputs ──────────────────────
    parser.add_argument(
        "--source", "-s",
        required=True,
        metavar="IMAGE_PATH",
        help="Path to the source image (contains the component you want to copy)."
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        metavar="IMAGE_PATH",
        help="Path to the target image (the component will be placed here)."
    )

    # ── Component ────────────────────────────
    parser.add_argument(
        "--component", "-c",
        default="armrest",
        metavar="COMPONENT_NAME",
        help=(
            "Name of the furniture component to swap. "
            "Examples: armrest, leg, seat, backrest, cushion. "
            "(default: armrest)"
        )
    )

    # ── Output ───────────────────────────────
    parser.add_argument(
        "--output", "-o",
        default="swapped_furniture.png",
        metavar="OUTPUT_PATH",
        help="File path for the final output image. (default: swapped_furniture.png)"
    )

    # ── Stable Diffusion options ─────────────
    sd_group = parser.add_argument_group("Stable Diffusion")
    sd_group.add_argument(
        "--sd-url",
        default="http://127.0.0.1:7860",
        metavar="URL",
        help="URL of the AUTOMATIC1111 WebUI API. (default: http://127.0.0.1:7860)"
    )
    sd_group.add_argument(
        "--denoising-strength",
        type=float,
        default=0.75,
        metavar="FLOAT",
        help=(
            "SD denoising strength between 0.0 and 1.0. "
            "Lower = subtle blend, higher = aggressive redraw. (default: 0.75)"
        )
    )
    sd_group.add_argument(
        "--sd-steps",
        type=int,
        default=30,
        metavar="INT",
        help="Number of SD sampling steps. (default: 30)"
    )

    # ── SAM options ──────────────────────────
    sam_group = parser.add_argument_group("Segment Anything Model (SAM)")
    sam_group.add_argument(
        "--sam-checkpoint",
        default="sam_vit_h_4b8939.pth",
        metavar="PATH",
        help=(
            "Path to the SAM model checkpoint (.pth file). "
            "(default: sam_vit_h_4b8939.pth in current directory)"
        )
    )
    sam_group.add_argument(
        "--sam-model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model variant to load. (default: vit_h)"
    )

    args = parser.parse_args()

    # ── Validation ───────────────────────────
    if not Path(args.source).is_file():
        parser.error(f"Source image not found: {args.source}")
    if not Path(args.target).is_file():
        parser.error(f"Target image not found: {args.target}")
    if not (0.0 <= args.denoising_strength <= 1.0):
        parser.error("--denoising-strength must be between 0.0 and 1.0")
    if args.sd_steps < 1:
        parser.error("--sd-steps must be at least 1")
    if not torch.cuda.is_available():
        parser.error(
            "CUDA GPU is required but not available. "
            "Please ensure NVIDIA drivers and CUDA toolkit are installed."
        )

    return args


# ─────────────────────────────────────────────
# STEP 1: Encode image to base64
# ─────────────────────────────────────────────
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────
# STEP 2: Use LLaVA to identify furniture & components
# ─────────────────────────────────────────────
def identify_furniture(image_path: str, image_number: int) -> dict:
    print(f"\nAnalyzing Image {image_number}: {Path(image_path).name}...")
    image_data = encode_image(image_path)

    prompt = """Analyze this furniture image and respond with all the furniture items and their components in the following JSON format:
{
  "furniture_items": [
    {
      "id": "item_1",
      "type": "Chair",
      "style": "Mission/Craftsman",
      "material": "Solid wood",
      "color": "Walnut brown",
      "components": ["seat", "backrest", "legs", "armrests"]
    }
  ]
}"""
    print("Prompt: ", prompt)
    response = ollama.chat(
        model="llava:latest",
        messages=[{"role": "user", "content": prompt, "images": [image_data]}]
    )

    raw_text = response["message"]["content"]
    print("Raw Response: ", raw_text)
    try:
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else {"furniture_items": []}
    except json.JSONDecodeError:
        parsed = {"furniture_items": []}

    return parsed


# ─────────────────────────────────────────────
# STEP 3: Load SAM (deferred, uses CLI args)
# ─────────────────────────────────────────────
def load_sam(checkpoint: str, model_type: str):
    from segment_anything import sam_model_registry
    print(f"\nLoading SAM ({model_type}) from {checkpoint}...")
    if not Path(checkpoint).is_file():
        print(f"ERROR: SAM checkpoint not found at '{checkpoint}'")
        print("Download it with:")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sys.exit(1)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to("cuda")
    print("SAM loaded on CUDA")
    return sam


# ─────────────────────────────────────────────
# STEP 4: Segment component with SAM
# ─────────────────────────────────────────────
def segment_component_with_sam(image_path: str, component_name: str, sam) -> np.ndarray:
    from segment_anything import SamAutomaticMaskGenerator
    print(f"\nSegmenting '{component_name}' from {Path(image_path).name}...")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_rgb)

    return select_mask_for_component(image_path, masks, component_name)


def select_mask_for_component(image_path: str, masks: list, component_name: str) -> np.ndarray:
    image = cv2.imread(image_path)

    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)[:5]
    mask_images = []

    for i, mask_data in enumerate(masks_sorted):
        overlay = image.copy()
        seg = mask_data["segmentation"].astype(np.uint8) * 255
        overlay[seg > 0] = [0, 255, 0]
        path = f"/tmp/mask_preview_{i}.jpg"
        cv2.imwrite(path, overlay)
        mask_images.append((i, path))

    print(f"Asking LLaVA to identify which mask contains '{component_name}'...")
    encoded_masks = [encode_image(p) for _, p in mask_images]

    prompt = f"""These images show different highlighted (green) regions of a furniture item.
Which image number (0 to {len(mask_images)-1}) best highlights the '{component_name}'?
Respond ONLY with the number, nothing else."""

    response = ollama.chat(
        model="llava:latest",
        messages=[{"role": "user", "content": prompt, "images": encoded_masks}]
    )

    try:
        idx = int(response["message"]["content"].strip())
        idx = max(0, min(idx, len(masks_sorted) - 1))
    except ValueError:
        idx = 0

    return masks_sorted[idx]["segmentation"].astype(np.uint8) * 255


def manual_roi_mask(image_path: str, component_name: str) -> np.ndarray:
    print(f"\nManual selection: Draw a box around the '{component_name}'")
    print("Click and drag to select, press ENTER to confirm, C to cancel")
    image = cv2.imread(image_path)
    roi = cv2.selectROI(f"Select {component_name}", image, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, roi)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return mask


# ─────────────────────────────────────────────
# STEP 5: Extract component pixels
# ─────────────────────────────────────────────
def extract_component(image_path: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    masked = cv2.bitwise_and(image, image, mask=mask)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return masked[y:y+h, x:x+w], mask[y:y+h, x:x+w]


# ─────────────────────────────────────────────
# STEP 6: Naive composite
# ─────────────────────────────────────────────
def naive_composite(
    target_path: str,
    component_img: np.ndarray,
    component_mask: np.ndarray,
    target_mask: np.ndarray
) -> np.ndarray:
    target = cv2.imread(target_path)
    coords = cv2.findNonZero(target_mask)
    x, y, w, h = cv2.boundingRect(coords)
    resized_comp = cv2.resize(component_img, (w, h))
    resized_mask = cv2.resize(component_mask, (w, h))
    roi = target[y:y+h, x:x+w]
    mask_3ch = cv2.merge([resized_mask, resized_mask, resized_mask]) / 255.0
    blended = (resized_comp * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    target[y:y+h, x:x+w] = blended
    return target


# ─────────────────────────────────────────────
# STEP 7: Stable Diffusion inpainting
# ─────────────────────────────────────────────
def sd_inpaint(
    target_path: str,
    inpaint_mask: np.ndarray,
    prompt: str,
    sd_url: str,
    denoising_strength: float,
    sd_steps: int
) -> np.ndarray:
    print("\nSending to Stable Diffusion for inpainting...")

    target_img = Image.open(target_path).convert("RGB")
    mask_img = Image.fromarray(inpaint_mask).convert("RGB")

    def pil_to_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    payload = {
        "init_images": [pil_to_b64(target_img)],
        "mask": pil_to_b64(mask_img),
        "prompt": prompt,
        "negative_prompt": "deformed, blurry, bad anatomy, extra limbs, watermark",
        "inpainting_fill": 1,
        "inpaint_full_res": True,
        "denoising_strength": denoising_strength,
        "steps": sd_steps,
        "cfg_scale": 7.5,
        "width": target_img.width,
        "height": target_img.height,
        "sampler_name": "DPM++ 2M Karras"
    }

    try:
        response = requests.post(f"{sd_url}/sdapi/v1/img2img", json=payload, timeout=120)
        response.raise_for_status()
        result_b64 = response.json()["images"][0]
        result_bytes = base64.b64decode(result_b64)
        result_img = Image.open(io.BytesIO(result_bytes))
        return np.array(result_img)
    except Exception as e:
        print(f"SD inpainting failed: {e}")
        print("Falling back to naive composite...")
        return None


# ─────────────────────────────────────────────
# STEP 8: Build SD inpaint prompt
# ─────────────────────────────────────────────
def build_inpaint_prompt(source_item: dict, target_item: dict, component: str) -> str:
    return (
        f"{source_item.get('style', 'modern')} style {component}, "
        f"{source_item.get('material', 'wood')} material, "
        f"{source_item.get('color', 'natural')} color, "
        f"attached to a {target_item.get('color', '')} {target_item['type']}, "
        f"seamlessly integrated, product photography, white background, "
        f"high quality, photorealistic"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  FURNITURE COMPONENT SWAP")
    print("═" * 60)
    print(f"  Source image   : {args.source}")
    print(f"  Target image   : {args.target}")
    print(f"  Component      : {args.component}")
    print(f"  Output         : {args.output}")
    print(f"  SD URL         : {args.sd_url}")
    print(f"  Denoising str  : {args.denoising_strength}")
    print(f"  SD steps       : {args.sd_steps}")
    print(f"  SAM checkpoint : {args.sam_checkpoint}")
    print(f"  SAM model type : {args.sam_model_type}")
    print(f"  Device         : CUDA (GPU)")
    print("═" * 60)

    # ── Identify furniture ───────────────────
    data1 = identify_furniture(args.source, 1)
    data2 = identify_furniture(args.target, 2)

    ollama.generate(model="llava:latest", prompt="", keep_alive=0)

    source_item = data1["furniture_items"][0]
    target_item = data2["furniture_items"][0]

    print(f"\nSource : {source_item['type']} ({source_item.get('color', 'unknown color')})")
    print(f"Target : {target_item['type']} ({target_item.get('color', 'unknown color')})")
    print(f"Swap   : {args.component}")

    # ── Load SAM ─────────────────────────────
    sam = load_sam(args.sam_checkpoint, args.sam_model_type)

    # ── Segment arm from source image ────────
    source_mask = segment_component_with_sam(args.source, args.component, sam)
    component_img, component_mask = extract_component(args.source, source_mask)

    cv2.imwrite("extracted_component.png", component_img)
    print("Extracted component saved: extracted_component.png")

    # ── Mark region in target image ──────────
    target_mask = segment_component_with_sam(args.target, args.component, sam)
    cv2.imwrite("target_mask.png", target_mask)
    print("Target mask saved: target_mask.png")

    # ── Stable Diffusion inpainting (always on) ──
    prompt = build_inpaint_prompt(source_item, target_item, args.component)
    print(f"\nSD Prompt: {prompt}")
    result = sd_inpaint(
        args.target,
        target_mask,
        prompt,
        args.sd_url,
        args.denoising_strength,
        args.sd_steps
    )

    if result is None:
        print("\nERROR: Stable Diffusion inpainting failed and no fallback is enabled.")
        print("Ensure AUTOMATIC1111 is running with --api at:", args.sd_url)
        sys.exit(1)

    # ── Save result ──────────────────────────
    if isinstance(result, np.ndarray):
        save_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR) if result.shape[2] == 3 else result
        cv2.imwrite(args.output, save_img)

    print(f"\n✓ Final image saved: {args.output}")


if __name__ == "__main__":
    main()