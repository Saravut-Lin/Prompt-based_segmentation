import os
import torch
import numpy as np
import gradio as gr
from PIL import Image

# Import SAM2 components
from sam2.sam2_image_predictor import SAM2ImagePredictor

############################################
# 1) MODEL LOADING (from checkpoint)
############################################
def load_model():
    checkpoint_path = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/sam2_pets_logs/checkpoints/checkpoint.pt"
    
    # Load checkpoint (assumes state_dict is stored under key 'model')
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model']
    
    # Rebuild the model architecture exactly as in training.
    # Build image encoder components
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    
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
    
    # Build memory attention components
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
    
    # Build memory encoder components
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
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
    
    # Build the SAM2 model using your training model class.
    from training.model.sam2 import SAM2Train
    model = SAM2Train(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=256,  # using ${scratch.resolution}
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
    
    # Load the checkpoint weights
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Wrap the model with SAM2ImagePredictor for interactive segmentation
    predictor = SAM2ImagePredictor(model)
    predictor._bb_feat_sizes = [(64, 64), (32, 32), (16, 16)]
    
    return predictor

print("Loading SAM2 model...")
predictor = load_model()
print("Model loaded successfully!")


############################################
# 2) POINT PROMPT LOGIC (Reset on Each Click)
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
# 3) BOX PROMPT LOGIC
############################################

box_start_point = None

def segment_with_box(image, evt: gr.SelectData):
    global box_start_point
    if image is None:
        return image
    
    import cv2
    if box_start_point is None:
        box_start_point = [evt.index[0], evt.index[1]]
        temp_img = image.copy()
        x, y = box_start_point
        cv2_radius = 5
        cv2_color = (0, 255, 0)
        cv2_thickness = -1
        temp_img = cv2.circle(temp_img, (x, y), cv2_radius, cv2_color, cv2_thickness)
        return temp_img
    
    end_point = [evt.index[0], evt.index[1]]
    box = [
        min(box_start_point[0], end_point[0]),
        min(box_start_point[1], end_point[1]),
        max(box_start_point[0], end_point[0]),
        max(box_start_point[1], end_point[1])
    ]
    box_start_point = None
    
    predictor.set_image(image)
    box_np = np.array(box)
    masks, scores, _ = predictor.predict(box=box_np, multimask_output=True)
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    
    colored_mask = np.zeros_like(image)
    colored_mask[..., 0] = mask * 255
    alpha = 0.5
    blended = (1 - alpha) * image + alpha * colored_mask
    return blended.astype(np.uint8)

def reset_box():
    global box_start_point
    box_start_point = None
    return "Box drawing reset. Click to start a new box."


############################################
# 4) SCRIBBLE PROMPT LOGIC
############################################

def process_mask_prompt(image, input_mask):
    predictor.set_image(image)
    input_mask_np = np.array(input_mask)
    masks, scores, _ = predictor.predict(mask_input=input_mask_np, multimask_output=True)
    best_idx = np.argmax(scores)
    return masks[best_idx]

def process_scribble(image, scribble):
    if image is None or scribble is None:
        return None
    scribble_gray = np.mean(scribble, axis=2)
    scribble_mask = (scribble_gray > 0).astype(np.float32)[..., None]
    mask = process_mask_prompt(image, scribble_mask)
    colored_mask = np.zeros_like(image)
    colored_mask[..., 0] = mask * 255
    alpha = 0.5
    blended = (1 - alpha) * image + alpha * colored_mask
    return blended.astype(np.uint8)


############################################
# 5) GRADIO UI CREATION
############################################

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# SAM2 Pet Segmentation App (Interactive)")
        gr.Markdown("Upload an image and click on it. Each click resets previous clicks. "
                    "When you click on the pet, it segments the pet; when you click on the background, it highlights the background automatically.")
        
        with gr.Tabs():
            # Point Prompts Tab
            with gr.TabItem("Point Prompts"):
                with gr.Row():
                    point_input_image = gr.Image(label="Input Image", type="numpy")
                    point_output_image = gr.Image(label="Segmentation Result", type="numpy")
                reset_points_btn = gr.Button("Reset Points")
                gr.Markdown("Click on the image. Each click is processed the pet or background based on its location.")
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
            
            # Box Prompts Tab
            with gr.TabItem("Box Prompts"):
                with gr.Row():
                    box_input_image = gr.Image(label="Input Image", type="numpy")
                    box_output_image = gr.Image(label="Segmentation Result", type="numpy")
                gr.Markdown("Click to set the start point, then click again to set the end point of the box")
                reset_btn = gr.Button("Reset Box")
                box_status = gr.Textbox(label="Box Status", value="Click to start drawing a box")
                box_input_image.select(
                    fn=segment_with_box,
                    inputs=[box_input_image],
                    outputs=box_output_image
                )
                reset_btn.click(
                    fn=reset_box,
                    inputs=[],
                    outputs=[box_status]
                )
            
            # Scribble Prompts Tab
            with gr.TabItem("Scribble Prompts"):
                with gr.Row():
                    scribble_input_image = gr.Image(label="Input Image", type="numpy")
                    scribble_canvas = gr.Image(label="Draw Scribble", type="numpy")
                    scribble_output_image = gr.Image(label="Segmentation Result", type="numpy")
                scribble_button = gr.Button("Segment with Scribble")
                
                def copy_image_to_canvas(image):
                    return image.copy() if image is not None else None
                
                scribble_input_image.change(
                    fn=copy_image_to_canvas,
                    inputs=[scribble_input_image],
                    outputs=[scribble_canvas]
                )
                scribble_button.click(
                    fn=process_scribble,
                    inputs=[scribble_input_image, scribble_canvas],
                    outputs=scribble_output_image
                )
                
    return demo

if __name__ == "__main__":
    import cv2  # required for box drawing
    demo = create_ui()
    demo.launch(share=True)
