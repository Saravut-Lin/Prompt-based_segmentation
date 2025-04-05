#!/usr/bin/env python3
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import logging
from pathlib import Path

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

def read_test_list(test_list_path):
    """Read the list of test images."""
    with open(test_list_path, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    return test_images

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate IoU, Dice coefficient, and pixel accuracy.
    """
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    total_pixels = pred_mask.size
    correct_pixels = (pred_mask == gt_mask).sum()
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    return {'iou': iou, 'dice': dice, 'pixel_accuracy': pixel_accuracy}

def visualize_results(image, gt_mask, pred_mask, metrics, output_path):
    """Create and save a visualization of the results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f"Prediction (IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f})")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(predictor, test_images, data_root, output_dir, visualization_samples=10):
    """Evaluate the model on the test images and compute metrics."""
    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    all_metrics = []
    class_metrics = {'pet': [], 'background': [], 'boundary': []}
    images_to_visualize = set(np.random.choice(
        len(test_images), min(visualization_samples, len(test_images)), replace=False
    ))
    
    for idx, image_name in enumerate(tqdm(test_images, desc="Evaluating")):
        img_path = os.path.join(data_root, "JPEGImages", image_name, "00000.jpg")
        mask_path = os.path.join(data_root, "Annotations", image_name, "00000.png")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Warning: Couldn't find image or mask for {image_name}, skipping.")
            continue
            
        image = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(mask_path))
        if gt_mask.max() > 1:
            gt_mask = gt_mask / 255
            
        # Use the predictor to compute the segmentation mask.
        predictor.set_image(image)
        h, w = image.shape[:2]
        prompt_point = np.array([[w // 2, h // 2]])
        prompt_label = np.array([1])
        
        masks, scores, logits = predictor.predict(
            point_coords=prompt_point,
            point_labels=prompt_label,
            multimask_output=True
        )
        
        best_mask_idx = np.argmax(scores)
        pred_mask = masks[best_mask_idx]
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        class_metrics['pet'].append(metrics['iou'])
        
        if idx in images_to_visualize:
            output_path = os.path.join(visualization_dir, f"{image_name}.png")
            visualize_results(image, gt_mask, pred_mask, metrics, output_path)
    
    avg_metrics = {
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'pixel_accuracy': np.mean([m['pixel_accuracy'] for m in all_metrics])
    }
    avg_class_metrics = {'pet': np.mean(class_metrics['pet']) if class_metrics['pet'] else 0}
    baseline_iou = 0.5 # Binary segmentation
    improvement = avg_metrics['iou'] - baseline_iou
    
    print("\n===== Evaluation Results =====")
    print(f"Average IoU: {avg_metrics['iou']:.4f}")
    print(f"Average Dice: {avg_metrics['dice']:.4f}")
    print(f"Average Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f}")
    print(f"Pet Category IoU: {avg_class_metrics['pet']:.4f}")
    print(f"Improvement over baseline: {improvement:.4f} ({improvement/baseline_iou*100:.2f}%)")
    
    results = {
        'average_metrics': avg_metrics,
        'class_metrics': avg_class_metrics,
        'baseline_comparison': {
            'baseline_iou': baseline_iou,
            'improvement': improvement,
            'percentage_improvement': improvement/baseline_iou*100
        },
        'per_image_metrics': {test_images[i]: all_metrics[i] for i in range(len(test_images))}
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 model on Oxford Pets dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--test-list', type=str, required=True, help='Path to test list file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--visualize', type=int, default=10, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    predictor = load_model(args.checkpoint)
    test_images = read_test_list(args.test_list)
    print(f"Found {len(test_images)} test images")
    
    results = evaluate_model(
        predictor=predictor,
        test_images=test_images,
        data_root=args.data_root,
        output_dir=args.output_dir,
        visualization_samples=args.visualize
    )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()


"""
python evaluate/evaluate_sam2_pets.py \
  --checkpoint /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/pets_logs/checkpoints/checkpoint.pt \
  --data-root /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets \
  --test-list /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets/test_list.txt \
  --output-dir /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/evaluate/evaluation_results_scratch_T_V_40
  
"""
