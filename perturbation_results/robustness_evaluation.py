import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import cv2
from skimage import util
import random
import logging
import pandas as pd

# Import SAM2 predictor and related modules
from sam2.sam2_image_predictor import SAM2ImagePredictor

#########################################
# Monkey-patch SAM2ImagePredictor.set_image
# Use .reshape() in place of .view() to avoid stride issues.
#########################################
_original_set_image = SAM2ImagePredictor.set_image

def patched_set_image(self, image):
    """
    Patched set_image method using .reshape(...) instead of .view(...).
    """
    self.reset_predictor()
    if isinstance(image, np.ndarray):
        logging.info("For numpy array image, we assume (HxWxC) format")
        self._orig_hw = [image.shape[:2]]
    elif hasattr(image, "size"):
        w, h = image.size
        self._orig_hw = [(h, w)]
    else:
        raise NotImplementedError("Image format not supported")
    input_image = self._transforms(image)
    input_image = input_image[None, ...].to(self.device)
    logging.info("Computing image embeddings for the provided image...")
    backbone_out = self.model.forward_image(input_image)
    _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
    feats = [
        # Replace .view(...) with .reshape(...)
        feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
    ][::-1]
    self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    self._is_image_set = True
    logging.info("Image embeddings computed.")

SAM2ImagePredictor.set_image = patched_set_image
#########################################
# End monkey patch
#########################################

def load_model(checkpoint_path):
    """
    Rebuild the SAM2 model architecture (matching your training configuration)
    and load the weights from checkpoint. Returns a SAM2ImagePredictor.
    """
    # 1) Load checkpoint (assumed to store state_dict in key 'model')
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model']
    
    # 2) Rebuild the model architecture exactly as in training.
    #    The parameters below are taken from your sam2_pets_finetune.yaml.
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    image_encoder = ImageEncoder(
        scalp=1,
        trunk=Hiera(embed_dim=112, num_heads=2, drop_path_rate=0.1),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest"
        )
    )
    
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.sam.transformer import RoPEAttention

    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            self_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[64, 64],
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1
            ),
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            cross_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[64, 64],
                rope_k_repeat=True,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64
            )
        ),
        num_layers=4
    )
    
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(num_pos_feats=64, normalize=True, scale=None, temperature=10000),
        mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
        fuser=Fuser(
            layer=CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True),
            num_layers=2
        )
    )
    
    # Build the SAM2 model using your training model class.
    # NOTE: Since your YAML does not specify backbone_stride, the default in SAM2Train is likely 16.
    # For image_size=256 and backbone_stride=16, sam_image_embedding_size = 256//16 = 16.
    from training.model.sam2 import SAM2Train
    model = SAM2Train(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=256,  # from ${scratch.resolution}
        # Do not override backbone_stride here so that it remains at its default (likely 16)
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
    )
    
    # 3) Load the state_dict into the model
    model.load_state_dict(state_dict)
    model = model.to('cpu')
    
    # 4) Wrap the model with SAM2ImagePredictor for evaluation
    predictor = SAM2ImagePredictor(model)
    # Override the predictor's _bb_feat_sizes to match the spatial resolution.
    # With image_size=256 and backbone_stride=16, we expect:
    #   sam_image_embedding_size = 256//16 = 16
    # Hence, set the last level to (16,16) and scale the others proportionally.
    predictor._bb_feat_sizes = [(64, 64), (32, 32), (16, 16)]
    return predictor

def load_test_dataset(data_dir, test_list_path=None):
    """Load the test dataset.

    This function assumes that the test list file contains subfolder names (e.g. "Abyssinian_1")
    and that images are located at:
      data_dir/JPEGImages/<folder_name>/00000.jpg
    and masks at:
      data_dir/Annotations/<folder_name>/00000.png
    """
    if test_list_path is None:
        test_list_path = os.path.join(data_dir, "test_list.txt")
    
    with open(test_list_path, 'r') as f:
        test_folders = [line.strip() for line in f.readlines()]
    
    images_dir = os.path.join(data_dir, "JPEGImages")
    masks_dir = os.path.join(data_dir, "Annotations")
    
    test_dataset = []
    for folder_name in test_folders:
        image_path = os.path.join(images_dir, folder_name, "00000.jpg")
        mask_path = os.path.join(masks_dir, folder_name, "00000.png")
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)
            test_dataset.append((image, mask))
        else:
            print(f"Warning: Could not find image or mask for {folder_name}")
    
    if len(test_dataset) == 0:
        raise ValueError("No valid test images found. Check your dataset paths and test list file.")
    
    return test_dataset

def apply_gaussian_noise(image, std_dev):
    """Apply Gaussian noise to the image."""
    image = np.array(image).astype(np.float32)
    noise = np.random.normal(0, std_dev, image.shape).astype(np.float32)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def apply_gaussian_blur(image, num_convolutions):
    """Apply Gaussian blur to the image."""
    image = np.array(image).astype(np.float32)
    # Define the 3x3 Gaussian kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16.0
    
    blurred_image = image.copy()
    for _ in range(num_convolutions):
        for c in range(blurred_image.shape[2]):
            blurred_image[:, :, c] = cv2.filter2D(blurred_image[:, :, c], -1, kernel)
    
    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred_image)

def apply_contrast_increase(image, factor):
    """Increase the contrast of the image."""
    image = np.array(image).astype(np.float32)
    contrasted_image = image * factor
    contrasted_image = np.clip(contrasted_image, 0, 255).astype(np.uint8)
    return Image.fromarray(contrasted_image)

def apply_contrast_decrease(image, factor):
    """Decrease the contrast of the image."""
    image = np.array(image).astype(np.float32)
    contrasted_image = image * factor
    contrasted_image = np.clip(contrasted_image, 0, 255).astype(np.uint8)
    return Image.fromarray(contrasted_image)

def apply_brightness_increase(image, value):
    """Increase the brightness of the image."""
    image = np.array(image).astype(np.float32)
    brightened_image = image + value
    brightened_image = np.clip(brightened_image, 0, 255).astype(np.uint8)
    return Image.fromarray(brightened_image)

def apply_brightness_decrease(image, value):
    """Decrease the brightness of the image."""
    image = np.array(image).astype(np.float32)
    darkened_image = image - value
    darkened_image = np.clip(darkened_image, 0, 255).astype(np.uint8)
    return Image.fromarray(darkened_image)

def apply_occlusion(image, size):
    """Apply occlusion to the image by replacing a square region with black pixels."""
    image = np.array(image).copy()
    if size == 0:
        return Image.fromarray(image)
    
    h, w, _ = image.shape
    x = random.randint(0, w - size - 1) if w > size else 0
    y = random.randint(0, h - size - 1) if h > size else 0
    
    image[y:y+size, x:x+size, :] = 0
    return Image.fromarray(image)

def apply_salt_and_pepper(image, amount):
    """Apply salt and pepper noise to the image."""
    image = np.array(image)
    if amount == 0:
        return Image.fromarray(image)
    
    noisy_image = util.random_noise(image / 255.0, mode='s&p', amount=amount, salt_vs_pepper=0.5)
    noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def compute_dice_score(pred_mask, gt_mask):
    """Compute the Dice score between predicted and ground truth masks."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    if union == 0:
        return 1.0
    else:
        return 2.0 * intersection / union

def evaluate_perturbation(predictor, test_dataset, perturbation_fn, perturbation_levels):
    """Evaluate model on perturbed dataset."""
    dice_scores = []
    
    for level in tqdm(perturbation_levels, desc="Evaluating perturbation"):
        level_dice_scores = []
        
        for image, gt_mask in tqdm(test_dataset, desc=f"Level {level}", leave=False):
            perturbed_image = perturbation_fn(image, level)
            
            predictor.set_image(perturbed_image)
            
            gt_mask_np = np.array(gt_mask)
            y_indices, x_indices = np.where(gt_mask_np > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
            
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
            
            masks, _, _ = predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            pred_mask = masks[0]
            dice_score = compute_dice_score(pred_mask, gt_mask_np > 0)
            level_dice_scores.append(dice_score)
        
        mean_dice = np.mean(level_dice_scores)
        dice_scores.append(mean_dice)
    
    return dice_scores

def create_plot(perturbation_name, perturbation_levels, dice_scores, save_dir="plots"):
    """Create and save a plot of perturbation level vs. Dice score."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(perturbation_levels, dice_scores, marker='o')
    plt.title(f"{perturbation_name} vs. Dice Score")
    plt.xlabel("Perturbation Level")
    plt.ylabel("Mean Dice Score")
    plt.grid(True)
    
    save_path = os.path.join(save_dir, f"{perturbation_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

def save_examples(image, perturbation_fn, perturbation_levels, save_dir="examples", perturbation_name=""):
    """Save examples of perturbed images."""
    if image is None:
        print(f"Cannot save examples for {perturbation_name}: No image provided")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(perturbation_levels), figsize=(20, 4))
    
    for i, level in enumerate(perturbation_levels):
        perturbed = perturbation_fn(image, level)
        axes[i].imshow(perturbed)
        axes[i].set_title(f"Level {level}")
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{perturbation_name.lower().replace(' ', '_')}_examples.png")
    plt.savefig(save_path)
    plt.close()

def main():
    # Load model
    checkpoint_path = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/pets_logs_T_V_40/checkpoints/checkpoint.pt"
    predictor = load_model(checkpoint_path)
    
    # Load test dataset
    data_dir = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets"
    test_list_path = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets/test_list.txt"
    test_dataset = load_test_dataset(data_dir, test_list_path)
    if not test_dataset:
        print("Error: No test images were loaded. Please check the dataset path and file structure.")
        return

    # Define perturbation levels for each type
    gaussian_noise_levels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    gaussian_blur_levels = list(range(10))  # 0 to 9 convolutions
    contrast_increase_levels = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.2, 1.25]
    contrast_decrease_levels = [1.0, 0.95, 0.9, 0.85, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    brightness_increase_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    brightness_decrease_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    occlusion_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    salt_pepper_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    
    perturbations = {
        "Gaussian Noise": (apply_gaussian_noise, gaussian_noise_levels),
        "Gaussian Blur": (apply_gaussian_blur, gaussian_blur_levels),
        "Contrast Increase": (apply_contrast_increase, contrast_increase_levels),
        "Contrast Decrease": (apply_contrast_decrease, contrast_decrease_levels),
        "Brightness Increase": (apply_brightness_increase, brightness_increase_levels),
        "Brightness Decrease": (apply_brightness_decrease, brightness_decrease_levels),
        "Occlusion": (apply_occlusion, occlusion_levels),
        "Salt and Pepper": (apply_salt_and_pepper, salt_pepper_levels)
    }
    
    results = {}
    for perturbation_name, (perturbation_fn, perturbation_levels) in perturbations.items():
        print(f"Evaluating {perturbation_name}")
        
        dice_scores = evaluate_perturbation(predictor, test_dataset, perturbation_fn, perturbation_levels)
        results[perturbation_name] = (perturbation_levels, dice_scores)
        
        create_plot(perturbation_name, perturbation_levels, dice_scores)
        
        example_img = test_dataset[0][0] if test_dataset else None
        save_examples(example_img, perturbation_fn, perturbation_levels, perturbation_name=perturbation_name)
        
        print(f"{perturbation_name} results:")
        for level, score in zip(perturbation_levels, dice_scores):
            print(f"  Level {level}: {score:.4f}")
    
    data = []
    for perturbation_name, (levels, scores) in results.items():
        for level, score in zip(levels, scores):
            data.append({
                "Perturbation": perturbation_name,
                "Level": level,
                "Dice Score": score
            })
    
    df = pd.DataFrame(data)
    df.to_csv("perturbation_results.csv", index=False)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()