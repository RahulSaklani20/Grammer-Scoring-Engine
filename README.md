# Grammar Scoring Engine

This project implements a Grammar Scoring Engine for spoken data samples. The model takes audio files as input and outputs a continuous score ranging from 0 to 5, predicting the grammar quality of the spoken content.

## Project Structure

```
.
├── Dataset/
│   ├── audios/          # Audio files
│   ├── train.csv        # Training data with labels
│   ├── test.csv         # Test data
│   └── sample_submission.csv
├── grammar_scoring_engine.py  # Main implementation
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Requirements

- Python 3.8+
- PyTorch
- Librosa
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your audio files in the `Dataset/audios/` directory
2. Ensure your training data is in `Dataset/train.csv` with columns: `filename,label`
3. Ensure your test data is in `Dataset/test.csv` with column: `filename`
4. Run the model:
```bash
python grammar_scoring_engine.py
```

The script will:
- Train the model on the training data
- Save the model weights to `grammar_scoring_model.pth`
- Generate predictions for the test set
- Create a submission file `submission.csv`

## Model Architecture

The model uses:
- MFCC (Mel-Frequency Cepstral Coefficients) features for audio representation
- A deep neural network with 4 fully connected layers
- ReLU activation and dropout for regularization
- Mean Squared Error loss function
- Adam optimizer

## Evaluation

The model is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Training progress is visualized in `training_history.png`. 