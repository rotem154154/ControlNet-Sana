# ControlNet Training

This repository contains code for training and fine-tuning a ControlNet model in a modular, maintainable, and professional manner.

## Repository Structure

```
controlnet_training/
├── config/
│   └── config.py          # Hyperparameters and configuration settings
├── data/
│   └── dataset.py         # Dataset and DataModule classes for handling HDF5 data
├── models/
│   └── finetuner.py       # Model definition (ControlNetFineTuner, CBlock, etc.)
├── scripts/
│   ├── encode_images.py   # Script to encode images and edges into an HDF5 file
│   └── encode_texts.py    # Script to encode text prompts into an HDF5 file
├── utils/
│   ├── helpers.py         # Utility functions (seeding, loss weighting, etc.)
│   └── logging.py         # (Optional) WandB logging setup
├── assets/                # Images, figures, and other media for documentation
├── train.py               # Main training script
├── inference.py           # Script for generating images after training
├── evaluate.py            # Script for computing evaluation metrics
├── requirements.txt       # Python dependencies
└── README.md              # Project instructions
```

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/controlnet_training.git
   cd controlnet_training
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data:**

   - Run the image encoding script:
     ```bash
     python scripts/encode_images.py
     ```
   - Run the text encoding script:
     ```bash
     python scripts/encode_texts.py
     ```

## Training

To train the model, run:

```bash
python train.py
```

## Inference

To generate images with the trained model, run:

```bash
python inference.py
```

## Evaluation

To compute evaluation metrics, run:

```bash
python evaluate.py
```

## Assets

All images and figures for documentation are stored in the `assets` folder.

<img src="assets/controlnet_images.jpg" alt="ControlNet Evaluation"/>
<img src="assets/controlnet_architecture.png" alt="ControlNet Architecture" width="600"/>
<img src="assets/controlnet_eval.png" alt="ControlNet Evaluation" height="100"/>

