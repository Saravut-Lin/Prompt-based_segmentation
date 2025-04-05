# Oxford Pets Segmentation using SAM2 Architecture

This repository contains an implementation of a pet segmentation application built with the SAM2 (Segment Anything Model 2) architecture. Unlike existing pre-trained segmentation models, this implementation presents a model **trained from scratch** specifically for pet segmentation using the SAM2 architecture as its foundation.

## Project Overview

The pet segmentation model was **built entirely from scratch** using the SAM2 architecture, trained on the Oxford Pets dataset. This approach allows for a specialized, domain-specific model that's optimized for identifying and segmenting pets in images without relying on any pre-trained weights.

Key features:
- **Custom-built model** trained from scratch on the Oxford Pets dataset
- Interactive segmentation with point and box prompts
- Clean and intuitive Gradio user interface
- Built on the core SAM2 architecture, but **trained completely from scratch** for pet-specific segmentation

## Model Architecture

This implementation uses the SAM2 architecture trained from zero on pet data. The model contains several key components:

### Image Encoder
The image encoder uses a Hiera-based backbone with the following configuration:
- Embedding dimension: 112
- Number of heads: 2
- Drop path rate: 0.1
- FPN neck with position encoding

### Memory Attention
A sophisticated attention mechanism with:
- Dimension model: 256
- 4 layers of memory attention
- RoPE (Rotary Position Encoding) attention for both self and cross-attention mechanisms

### Memory Encoder
Processes masks into memory representations with:
- Output dimension: 64
- Position encoding with sine functions
- Mask downsampling with kernel size 3
- Fusion using CXBlock layers

### Mask Decoder
The SAM2 mask decoder generates the final segmentation masks with:
- Multiple mask output capabilities
- IoU prediction for mask quality assessment
- Object score prediction

These components work together to enable accurate pet segmentation from minimal user input. The model was **trained completely from scratch** using the pet dataset - it does not use or depend on any pre-trained weights.

## User Interface

The application provides two key interaction methods:

### Point Prompts
- Each click is automatically classified as foreground (pet) or background
- Points are processed individually based on their location
- The application analyzes a small patch around each click to determine if it's part of a pet or background
- Points are used to refine the segmentation in challenging areas
- Visualization includes a colored overlay showing the current segmentation

### Box Prompts
- Two-click approach where you define a bounding box around the pet
- First click sets the top-left corner (shown with a green dot)
- Second click sets the bottom-right corner
- If the box contains pet regions (coverage > 0), it's colored green
- If the box contains only background regions, it's colored red
- Segmentation is restricted to the box region
- The bounding box is drawn in the final result for clarity

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pet-segmentation.git
cd pet-segmentation

# Install dependencies 
pip install -r requirements.txt

# Optional: Install CUDA requirements for GPU acceleration
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## Running the Application

```bash
python UI/interactive_segmentation.py
```

This will launch the Gradio web interface where you can upload images and interactively segment pets using either point or box prompts.

## Implementation Details

The repository is organized as follows:

- `UI/`: Contains the interactive Gradio interface and application logic
  - `interactive_segmentation.py`: Main application with the Gradio interface
  - `abstract.py`: Abstract class implementations

- `sam2/`: The core model architecture components, **built and trained from scratch**
  - `modeling/`: Core model components including backbones, encoders, and decoders
  - `utils/`: Utility functions for image processing and model operations
  - `sam2_image_predictor.py`: Class for making predictions on images
  - `sam2_video_predictor.py`: Extended predictor for video sequences

- `training/`: Training modules used to build the model from scratch on the Oxford Pets dataset
  - `dataset/`: Data loading and augmentation utilities
  - `model/`: Model definition for training
  - `utils/`: Training utilities including loggers and distributed training tools
  - `loss_fns.py`: Loss functions for training
  - `optimizer.py`: Optimizer configurations
  - `trainer.py`: Main training loop implementation
  - `train.py`: Script for launching training jobs

The SAM2 architecture includes components like:

1. **Image Encoder**: Processes the input image into a feature representation using a Hiera backbone
2. **Memory Attention**: Manages attention between current inputs and previous states
3. **Memory Encoder**: Encodes mask information into memory representations
4. **Prompt Encoder**: Processes user input (clicks or boxes) to guide segmentation
5. **Mask Decoder**: Generates the final segmentation masks

## Technical Approach

This project demonstrates how the SAM2 architecture can be **trained entirely from scratch** for a specific segmentation task. By training on the Oxford Pets dataset, we've created a specialized model that excels at pet segmentation while maintaining the interactive capabilities of the SAM2 approach.

Key technical aspects:
- **Ground-up training**: The model was trained from scratch on pet data
- Optimization for pet-specific segmentation
- No reliance on pre-trained weights or knowledge transfer
- Maintained interactive capabilities from the original SAM2 architecture
- Point sampling strategy for interactive refinement
- Box-guided segmentation for targeted regions

### Training Methodology

The model was trained using:
- Batch size: 1 (due to CPU training constraints)
- Resolution: 256x256
- Optimizer: AdamW with learning rate 5.0e-6 for vision backbone
- Training epochs: 40
- Loss functions: 
  - Focal loss for mask prediction
  - Dice loss for boundary accuracy
  - IoU loss for segmentation quality

## Data

The model was trained using the Oxford Pets dataset, which includes:
- 37 pet categories (different breeds of cats and dogs)
- ~200 images per class (approximately 7,400 images total)
- Pet segmentation annotations with pixel-level masks
- Varied poses, lighting conditions, and backgrounds
- Train/validation splits provided in the dataset

## Performance Optimization

The implementation includes several optimizations:
- Optional hole filling in predicted masks (configured by `fill_hole_area`)
- Non-overlapping constraints for multi-object segmentation
- Memory management for efficient processing
- Customizable threshold for mask binary decisions
- Support for both CPU and GPU inference

## Command Line Options

The application supports several command-line options for advanced usage:

```
python UI/interactive_segmentation.py --checkpoint_path /path/to/checkpoint.pt
```

## Extending the Project

This implementation can be extended in several ways:
1. Training on additional pet categories or animal species
2. Adding support for video segmentation using the `sam2_video_predictor.py`
3. Implementing additional prompting methods beyond points and boxes
4. Optimizing for mobile deployment

## Conclusion

This implementation showcases how the SAM2 architecture can be **built from scratch** for specialized segmentation tasks. By training specifically on pet data from the ground up, we've created an interactive and effective pet segmentation tool while maintaining the responsive and intuitive interface that makes SAM2 powerful.

The ability to train a specialized segmentation model from scratch demonstrates the flexibility of the SAM2 architecture and its applicability to domain-specific problems without relying on pre-trained weights.

## Acknowledgments

This implementation's architecture is based on the SAM2 design, but the model itself was **built and trained entirely from scratch** on the Oxford Pets dataset. No pre-trained weights or transfer learning were utilized. The implementation benefits from the contributions of the SAM2 architecture design, the Oxford Pets dataset creators, and the open-source ML community.
