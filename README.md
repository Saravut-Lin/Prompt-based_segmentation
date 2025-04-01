# SAM-2 Training on Oxford-IIIT Pet Dataset

This repository contains my implementation of fine-tuning SAM 2 (Segment Anything in Images and Videos) on the Oxford-IIIT Pet Dataset. SAM 2 is a foundation model for promptable visual segmentation in images and videos, and this project demonstrates how to adapt it to a specific domain using a structured pet dataset.

## Overview

SAM 2 is a powerful segmentation model that can segment objects in images and videos based on prompts like points, boxes, or masks. This training framework enables researchers and practitioners to adapt SAM 2 to their specific domains and use cases.

The repository is organized as follows:
- **training/**: Core training code
  - **dataset/**: Dataset and dataloader classes and transforms
  - **model/**: Model implementation including SAM2Train for training/fine-tuning
  - **utils/**: Training utilities for logging, distributed training, etc.
  - **scripts/**: Helper scripts such as video frame extraction
  - **loss_fns.py**: Loss functions for training
  - **optimizer.py**: Optimizer utilities with scheduler support
  - **trainer.py**: Main trainer class implementing train/eval loops
  - **train.py**: Script to launch training jobs
- **sam2/**: Core model implementation
  - **modeling/**: Model architecture components
  - **utils/**: Utilities for inference and post-processing
  - **build_sam.py**: Model loading and building utilities

## Prerequisites

- PyTorch (>= 2.5.1)
- torchvision (>= 0.20.1)
- CUDA-capable GPU recommended for full-scale training
- Additional dependencies listed in setup.py

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/Saravut-Lin/Prompt-based_segmentation.git
cd Prompt-based_segmentation
pip install -e ".[dev]"
```

## Dataset Preparation

The Oxford-IIIT Pet Dataset was converted to a format compatible with the SAM 2 training framework using create_dataset2.py. This script transforms the original dataset structure into the format expected by SAM 2.

### Original Oxford-IIIT Pet Dataset Structure

The original dataset has the following structure:
- **images/** - Contains all pet images in a flat directory
- **annotations/trimaps/** - Contains trimap segmentation masks (1: Foreground, 2: Background, 3: Not classified)
- **list.txt** - List of all images with class information
- **trainval.txt & test.txt** - Train/validation/test splits

### Transformed Dataset Structure for SAM 2

The create_dataset2.py script converts this into:

```
data/oxford_pets/
├── JPEGImages/
│   ├── Abyssinian_1/
│   │   └── 00000.jpg
│   ├── american_bulldog_203/
│   │   └── 00000.jpg
│   └── ...
├── Annotations/
│   ├── Abyssinian_1/
│   │   └── 00000.png
│   ├── american_bulldog_203/
│   │   └── 00000.png
│   └── ...
├── train_list.txt
├── val_list.txt
├── train_list_tiny.txt
└── val_list_tiny.txt
```

In this structure:
- Each pet image is treated as a single-frame "video"
- Each breed/ID becomes a folder name (e.g., "Abyssinian_1")
- The trimap masks are converted to binary masks (foreground/background)
- Train/val splits are created (80/20) with tiny subsets for testing

## Training

### 1. Configure training

The configuration for Oxford Pets is already prepared in sam2_pets_finetune_no_checkpoint.yaml:

```yaml
dataset:
  # Paths to Dataset - update these to your local paths
  img_folder: /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets/JPEGImages
  gt_folder: /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets/Annotations
  train_file_list_txt: /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets/train_list_tiny.txt
  val_file_list_txt: /Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/data/oxford_pets/val_list_tiny.txt
```

### 2. Run training

For single-node training with 8 GPUs:

```bash
python training/train.py \
  -c configs/your_config.yaml \
  --use-cluster 0 \
  --num-gpus 8
```

For multi-node distributed training with SLURM:

```bash
python training/train.py \
  -c configs/your_config.yaml \
  --use-cluster 1 \
  --num-gpus 8 \
  --num-nodes 2 \
  --partition $PARTITION \
  --qos $QOS \
  --account $ACCOUNT
```

### 3. Monitor training

Training logs and checkpoints are saved to the specified output directory:

```bash
tensorboard --logdir sam2_logs/your_config/tensorboard
```

## Fine-tuning on Oxford-IIIT Pet Dataset

I've provided a configuration file for fine-tuning SAM 2 on the Oxford Pets dataset. This configuration is optimized to run on CPU or with limited GPU resources.

```bash
python training/train.py \
  -c sam2_pets_finetune_no_checkpoint.yaml \
  --use-cluster 0 \
  --num-gpus 1  # Set to 0 for CPU training
```

The configuration includes:
- Reduced resolution (256x256) for faster training
- Small batch size suitable for CPU/limited GPU
- Using point prompts for training with iterative correction
- Options for training from scratch or fine-tuning from pretrained weights

For faster iteration during development, you can use the tiny dataset splits:

```yaml
dataset:
  train_file_list_txt: /path/to/oxford_pets/train_list_tiny.txt
  val_file_list_txt: /path/to/oxford_pets/val_list_tiny.txt
```

## Implementation Details

### Dataset Transformation Script

The create_dataset2.py script performs the following operations:
1. Reads images from the original Oxford-IIIT Pet dataset
2. Creates the SAM 2 compatible directory structure
3. Converts trimap segmentations to binary masks
4. Creates train/validation splits
5. Creates small subsets for testing

```python
# Example from create_dataset2.py
for img_path in image_files:
    # Get base filename
    basename = os.path.basename(img_path)
    filename = os.path.splitext(basename)[0]
    
    # Create directories for this "video"
    video_dir = os.path.join(OUT_IMAGES, filename)
    mask_dir = os.path.join(OUT_MASKS, filename)
    
    # Convert trimap to binary mask
    trimap = np.array(Image.open(trimap_path))
    binary_mask = np.zeros_like(trimap, dtype=np.uint8)
    binary_mask[trimap == 1] = 1  # foreground
    binary_mask[trimap == 3] = 1  # boundary
```

### Training Configuration

Our configuration treats each pet image as a single image for training:

```yaml
# Key parameters in our configuration
scratch:
  resolution: 256            # Reduced resolution for quicker CPU training
  train_batch_size: 2
  num_frames: 1              # Single image mode
  max_num_objects: 1         # One pet per image
  
model:
  # Training parameters
  prob_to_use_pt_input_for_train: 1.0  # Always use point prompts
  prob_to_use_box_input_for_train: 0.0
  num_frames_to_correct_for_train: 1   # Single image mode
  num_correction_pt_per_frame: 3       # 3 correction points per frame
```

## Using the Fine-tuned Model

After training, you can use the fine-tuned model for pet segmentation:

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image

# Load the fine-tuned model
model = build_sam2("sam2_pets_finetune_no_checkpoint.yaml", 
                   "sam2_pets_logs/checkpoints/checkpoint.pt")
predictor = SAM2ImagePredictor(model)

# Load a test image
image = np.array(Image.open("path/to/test_pet.jpg"))
predictor.set_image(image)

# Single point prompt in the center of the pet
h, w = image.shape[:2]
point_coords = np.array([[w//2, h//2]])  # Center point
point_labels = np.array([1])  # Positive point

# Get prediction
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)
```

The fine-tuned model will be more effective at segmenting pets compared to the general-purpose SAM 2 model, requiring fewer prompts to achieve accurate segmentation.

## Results

The fine-tuned model demonstrates improved segmentation quality on pet images, requiring fewer user prompts to achieve accurate masks. Key improvements include:
1. Better handling of complex fur patterns and pet anatomical features
2. Improved boundary accuracy around ears, paws, and tails
3. More robust segmentation with just a single point prompt
4. Better distinction between similar-looking breeds

## Evaluation and Comparison

The script evaluate_sam2_pets.py is used to quantitatively and qualitatively evaluate the segmentation performance on the testing set. In this section, the following points are considered:
- **Metrics**: Evaluation metrics include Intersection over Union (IoU), Dice coefficient (F1 score for segmentation), and Pixel Accuracy as appropriate for the dataset.
- **Baseline Comparison**: The baseline's mean IoU across the three categories is 0.33, and the objective is for the best-performing method to outperform this baseline.

To run the evaluation, execute:

```bash
python evaluate_sam2_pets.py --config <config_file> --test_data <test_dataset_path>
```

Review the generated metrics and visual comparisons to assess both the quantitative performance and qualitative segmentation quality.

## User Interface

The script interactive_segmentation.py implements a simple user interface that connects with the trained point-based segmentation model, enabling the development of a small segmentation application. Key features include:
- **Interactive Prompting**: After loading an image, users can click on an object, and the network predicts the object mask, which is then visualized in the interface.
- **Additional Prompts**: The interface also supports other forms of prompts (e.g., bounding box or text) to guide the segmentation.

To launch the interactive segmentation tool, run:

```bash
python interactive_segmentation.py --model_checkpoint path/to/checkpoint.pt --image path/to/image.jpg
```

Follow the on-screen instructions to interactively segment objects in your images.

## Robustness Exploration Details

The script robustness_evaluation.py is designed to assess the robustness of the segmentation model against various perturbations without retraining the model. The evaluation involves:
- **Perturbations**: A series of eight perturbations are applied to the test images. For each perturbation, the segmentation accuracy (measured by the Dice score) is re-calculated, and a plot is produced showing segmentation accuracy versus the amount of perturbation.
- **Evaluation Metric**: The primary metric is the mean Dice accuracy on the test set, evaluated under increasing levels of perturbation.

The eight perturbations include:

### a) Gaussian Pixel Noise:
- Add a Gaussian distributed random number to each pixel with increasing standard deviations from {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}.
- Ensure pixel values remain integers in the range 0–255 (e.g., replace negatives with 0 and values >255 with 255).

### b) Gaussian Blurring:
- Create test images by convolving the original image with a 3x3 mask repeatedly (0, 1, 2, …, 9 times), which approximates Gaussian blurring with increasing standard deviation.

### c) Image Contrast Increase:
- Multiply each pixel by factors {1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25}, ensuring pixel values are clamped between 0 and 255.

### d) Image Contrast Decrease:
- Multiply each pixel by factors {1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10}.

### e) Image Brightness Increase:
- Add values {0, 5, 10, 15, 20, 25, 30, 35, 40, 45} to each pixel, clamping values above 255.

### f) Image Brightness Decrease:
- Subtract values {0, 5, 10, 15, 20, 25, 30, 35, 40, 45} from each pixel, ensuring no value goes below 0.

### g) Occlusion of the Image Increase:
- Replace a randomly placed square region of the image with black pixels. The square edge length varies over {0, 5, 10, 15, 20, 25, 30, 35, 40, 45}.

### h) Salt and Pepper Noise:
- Add salt and pepper noise with increasing strength, where the noise amount is varied over {0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18}.

Each perturbation generates its own plot of segmentation accuracy versus perturbation level along with example images. To run the robustness evaluation, execute:

```bash
python robustness_evaluation.py --config <config_file> --test_data <test_dataset_path>
```

Analyze the resulting plots to understand how various perturbations affect the model's performance.

## Citations

If you use this code, please cite both SAM 2 and the Oxford-IIIT Pet Dataset:

```
@misc{sam2,
  author = {Meta AI},
  title = {SAM 2: Segment Anything in Images and Videos},
  year = {2024},
  url = {https://github.com/facebookresearch/sam2}
}

@inproceedings{parkhi12a,
  author = {Parkhi, O. M. and Vedaldi, A. and Zisserman, A. and Jawahar, C.~V.},
  title = {Cats and Dogs},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2012}
}
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
