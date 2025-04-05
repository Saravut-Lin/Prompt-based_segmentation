import os
import torch
import numpy as np
import gradio as gr
from PIL import Image

# Import SAM2 components
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1) MODEL LOADING (from checkpoint)
def load_model():
    checkpoint_path = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/pets_logs_T_V_40/checkpoints/checkpoint.pt"
    
    # Load checkpoint (assumes state_dict is stored under key 'model')
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model']
    
    # Rebuild the model architecture exactly as in training.
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.sam.transformer import RoPEAttention
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
    from training.model.sam2 import SAM2Train
    
    image_encoder = ImageEncoder(
        scalp=1,
        trunk=Hiera(embed_dim=112, num_heads=2, drop_path_rate=0.1),
        neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256, normalize=True, scale=None, temperature=10000
            ),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest"
        )
    )
    
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
    
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64, normalize=True, scale=None, temperature=10000
        ),
        mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
        fuser=Fuser(
            layer=CXBlock(
                dim=256, kernel_size=7, padding=3,
                layer_scale_init_value=1e-6, use_dwconv=True
            ),
            num_layers=2
        )
    )
    
    model = SAM2Train(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=256,
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
    
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Wrap with predictor
    predictor = SAM2ImagePredictor(model)
    predictor._bb_feat_sizes = [(64, 64), (32, 32), (16, 16)]
    return predictor

print("Loading SAM2 model...")
predictor = load_model()
print("Model loaded successfully!")


############################################
# 2) POINT PROMPT LOGIC
############################################

# Global default mask (computed once)
default_mask = None

def compute_default_mask(image):
    """Compute a reference segmentation mask using a full-image bounding box."""
    predictor.set_image(image)
    h, w = image.shape[:2]
    full_box = np.array([0, 0, w-1, h-1])
    masks_box, scores_box, _ = predictor.predict(box=full_box, multimask_output=True)
    best_idx_box = np.argmax(scores_box)
    return masks_box[best_idx_box]

def reset_points():
    """Clear any stored default mask."""
    global default_mask
    default_mask = None
    return None

def process_point_prompt(image, point, label):
    """
    Process a single point click.
    """
    predictor.set_image(image)
    point_coords = np.array([point])
    point_labels = np.array([label])
    
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    
    # If the click is determined as background, invert the mask.
    if label == 0:
        mask = 1 - mask
    return mask

def segment_with_point(image, evt: gr.SelectData, patch_size=5, threshold=0.5):
    """
    When a point is clicked, automatically determine its label by examining a patch in the default mask.
    Each click resets previous clicks.
    """
    global default_mask
    if image is None:
        return None
    
    # Compute default mask if not already computed
    if default_mask is None:
        default_mask = compute_default_mask(image)
    
    # Get current click coordinates
    x, y = evt.index[0], evt.index[1]
    # Define a patch around the click
    half_patch = patch_size // 2
    y_min = max(0, y - half_patch)
    y_max = min(default_mask.shape[0], y + half_patch + 1)
    x_min = max(0, x - half_patch)
    x_max = min(default_mask.shape[1], x + half_patch + 1)
    patch = default_mask[y_min:y_max, x_min:x_max]
    
    # Automatically determine label: foreground if the patch average > threshold, else background
    label = 1 if np.mean(patch) > threshold else 0
    
    # Process only this click (reset accumulated clicks each time)
    mask = process_point_prompt(image, [x, y], label)
    
    # Create an overlay for visualization
    colored_mask = np.zeros_like(image)
    colored_mask[..., 0] = mask * 255
    alpha = 0.5
    blended = (1 - alpha) * image + alpha * colored_mask
    return blended.astype(np.uint8)


############################################
# 3) BOX PROMPT LOGIC (two-click approach)
############################################
#
#  - If coverage_fraction > 0 => pet segmentation in the box (GREEN)
#  - If coverage_fraction == 0 => background in the box (RED)
#  - Segmentation is clamped to the bounding box region.

box_start_point = None

def segment_with_box(image, evt: gr.SelectData):
    """
    Two-click approach:
      - First click => store start point (draw green dot)
      - Second click => finalize bounding box
        * coverage_fraction = mean of default_mask in box
        * if coverage_fraction > 0 => pet segmentation (green)
        * else => background (red)
    """
    global default_mask, box_start_point
    if image is None:
        return None
    
    if default_mask is None:
        default_mask = compute_default_mask(image)
    
    import cv2
    
    # If no start point => record first click
    if box_start_point is None:
        box_start_point = [evt.index[0], evt.index[1]]
        temp_img = image.copy()
        x, y = box_start_point
        cv2.circle(temp_img, (x, y), radius=5, color=(0,255,0), thickness=-1)
        return temp_img
    
    # Second click => do bounding box
    end_point = [evt.index[0], evt.index[1]]
    xmin = min(box_start_point[0], end_point[0])
    ymin = min(box_start_point[1], end_point[1])
    xmax = max(box_start_point[0], end_point[0])
    ymax = max(box_start_point[1], end_point[1])
    
    # Reset for the next new box
    box_start_point = None
    
    if xmin == xmax or ymin == ymax:
        return image  # basically no box
    
    # Check coverage
    patch = default_mask[ymin:ymax, xmin:xmax]
    coverage_fraction = np.mean(patch)
    
    h, w = image.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.float32)
    
    if coverage_fraction > 0:
        # => Some pet in the box => do pet segmentation
        predictor.set_image(image)
        box_np = np.array([xmin, ymin, xmax, ymax])
        masks, scores, _ = predictor.predict(box=box_np, multimask_output=True)
        best_idx = np.argmax(scores)
        raw_mask = masks[best_idx]
        
        # clamp to the box region
        final_mask[ymin:ymax, xmin:xmax] = raw_mask[ymin:ymax, xmin:xmax]
        
        box_color = (0, 255, 0)  # green
        color_idx = 1           # green channel
    else:
        # => No pet => background
        final_mask[ymin:ymax, xmin:xmax] = 1.0
        
        box_color = (0, 0, 255)  # red
        color_idx = 2           # red channel
    
    # Create overlay
    colored_mask = np.zeros_like(image)
    colored_mask[..., color_idx] = (final_mask * 255).astype(np.uint8)
    
    alpha = 0.5
    blended = (1 - alpha) * image + alpha * colored_mask
    
    # Draw bounding box
    cv2.rectangle(blended, (xmin, ymin), (xmax, ymax), box_color, thickness=2)
    
    return blended.astype(np.uint8)

def reset_box():
    """Reset the segmentation result and the stored start point."""
    global box_start_point
    box_start_point = None
    return None, "Box drawing reset. Click first point, then second point."


############################################
# 4) GRADIO UI CREATION (NO SCRIBBLE)
############################################

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Pet Segmentation App (Points & Boxes Only)")
        gr.Markdown(
            "- **Point Prompts**: Each click is automatically considered pet or background based on a patch.\n"
            "- **Box Prompts**: Two-click approach (Segmentation is restricted to that box region).\n"
            "  1) First click = top-left corner (green dot)\n"
            "  2) Second click = bottom-right corner.\n"
        )

        with gr.Tabs():
            # Point Prompts
            with gr.TabItem("Point Prompts"):
                with gr.Row():
                    point_input_image = gr.Image(label="Input Image", type="numpy")
                    point_output_image = gr.Image(label="Segmentation Result", type="numpy")
                reset_points_btn = gr.Button("Reset Points")
                gr.Markdown("Click on the image. Each click is processed individually based on its location.")
                point_input_image.select(
                    fn=segment_with_point,
                    inputs=[point_input_image],
                    outputs=point_output_image
                )
                reset_points_btn.click(
                    fn=reset_points,
                    inputs=[],
                    outputs=point_output_image
                )

            # Box Prompts (Two-click approach)
            with gr.TabItem("Box Prompts"):
                with gr.Row():
                    box_input_image = gr.Image(label="Input Image", type="numpy")
                    box_output_image = gr.Image(label="Segmentation Result", type="numpy")
                box_status = gr.Textbox(label="Box Status", value="Click first point, then second point.")
                reset_box_btn = gr.Button("Reset Box")
                
                gr.Markdown(
                    "1) **First click**: sets the top-left corner (shown with green dot)\n"
                    "2) **Second click**: sets the bottom-right corner.\n"
                    "   - If coverage > 0 ⇒ box is **green** (pet)\n"
                    "   - Else ⇒ box is **blue** (background)\n"
                    "Segmentation is restricted to the box. The bounding box is drawn in the final result."
                )
                
                box_input_image.select(
                    fn=segment_with_box,
                    inputs=[box_input_image],
                    outputs=box_output_image
                )
                reset_box_btn.click(
                    fn=reset_box,
                    inputs=[],
                    outputs=[box_output_image, box_status]
                )

    return demo

if __name__ == "__main__":
    import cv2  # used for drawing rectangles in the box prompt
    demo = create_ui()
    demo.launch(share=True)