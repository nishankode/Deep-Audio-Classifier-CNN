# Capuchin Bird Audio Classification

This project tackles the famous HP Challenge by building a deep learning model to detect and count capuchin bird calls from audio recordings. The solution converts raw audio files (in WAV and MP3 formats) into spectrograms, builds a TensorFlow dataset, trains a convolutional neural network (CNN) on the spectrograms, and finally makes predictions on unseen forest recordings.

Challenge Dataset Link: https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation and Requirements](#installation-and-requirements)
- [Project Structure](#project-structure)
- [Code Walkthrough](#code-walkthrough)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Visualization of Audio Waves](#2-visualization-of-audio-waves)
  - [3. Building the TensorFlow Dataset](#3-building-the-tensorflow-dataset)
  - [4. Spectrogram Generation](#4-spectrogram-generation)
  - [5. Model Building and Training](#5-model-building-and-training)
  - [6. Evaluation and Plotting Metrics](#6-evaluation-and-plotting-metrics)
  - [7. Making Predictions](#7-making-predictions)
  - [8. Final Inference and Saving Results](#8-final-inference-and-saving-results)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project demonstrates the end-to-end process of classifying capuchin bird calls in audio recordings using deep learning. The key steps include:
- Loading and processing raw audio files
- Converting audio into spectrograms
- Building a custom TensorFlow dataset for training and testing
- Constructing a CNN model for binary classification (capuchin vs. non-capuchin audio)
- Evaluating model performance through loss, recall, and precision metrics
- Making predictions on both single clips and larger forest recordings
- Saving the trained model and prediction results for further analysis

The challenge (originally part of the HP Unlocked Challenge 3) provided the basis for using advanced audio processing techniques along with deep learning to count and identify bird calls in natural forest recordings.

---

## Dataset

The dataset used for this project is derived from the HP Challenge and consists of two main folders:
- `Parsed_Capuchinbird_Clips`: Contains audio files with capuchin bird calls.
- `Parsed_Not_Capuchinbird_Clips`: Contains audio files of background or non-capuchin sounds.

Additionally, a separate folder (e.g., `Forest Recordings`) includes longer recordings from which multiple clips are sliced for complete inference.

---

## Installation and Requirements

Ensure that you have Python 3.6+ installed along with the following libraries:

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow I/O](https://www.tensorflow.org/io)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Warnings](https://docs.python.org/3/library/warnings.html) (standard library)

You can install the Python packages using pip:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
├── data
│   ├── Parsed_Capuchinbird_Clips
│   ├── Parsed_Not_Capuchinbird_Clips
│   └── Forest Recordings
├── notebooks
│   └── HP_Challenge_Exploration.ipynb
├── Output
│   ├── Model
│   │   └── capuchin_best_model.h5
│   └── Result
│       └── Result.csv
├── README.md
└── main.py
```

- **data/**: Contains the audio files organized in subfolders.
- **notebooks/**: (Optional) Jupyter Notebook for exploratory analysis.
- **Output/**: Folder where the trained model and final prediction results are saved.
- **main.py**: Main Python script that runs the entire pipeline.

---

## Code Walkthrough

### 1. Data Loading and Preprocessing

The project starts by importing necessary libraries (os, matplotlib, TensorFlow, NumPy, Pandas, etc.). The code loads WAV files from the specified directories and defines functions to convert audio into 16 kHz mono signals.

### 2. Visualization of Audio Waves

For an initial visual inspection, the code plots the waveforms of both capuchin and non-capuchin audio samples using Matplotlib.

### 3. Building the TensorFlow Dataset

File paths for positive (capuchin) and negative (non-capuchin) samples are defined. TensorFlow’s `tf.data.Dataset` API is used to list, label, and concatenate the audio file paths, creating a unified dataset for training.

### 4. Spectrogram Generation

A preprocessing function is created to:
- Load the audio file using TensorFlow I/O.
- Truncate or zero-pad the waveform to a fixed length (e.g., 48000 samples).
- Convert the waveform into a spectrogram using the short-time Fourier transform (STFT).
- Expand dimensions to prepare the data for the CNN.

### 5. Model Building and Training

A Convolutional Neural Network (CNN) is constructed using TensorFlow and Keras. The architecture includes:
- Two Conv2D layers with ReLU activation.
- A MaxPooling2D layer for dimensionality reduction.
- A Flatten layer followed by Dense layers leading to a final sigmoid activation for binary classification.

The model is compiled with the Adam optimizer, binary cross-entropy loss, and metrics such as recall and precision. It is then trained for several epochs on the training dataset with validation on a test split.

### 6. Evaluation and Plotting Metrics

The training history is used to plot:
- Loss and validation loss over epochs.
- Recall and precision metrics for both training and validation sets.
These plots help assess the model’s performance and monitor any overfitting.

### 7. Making Predictions

The trained model is applied to a batch from the test dataset. Predictions are converted to binary classes (using a threshold of 0.5) and compared with true labels to evaluate accuracy.

### 8. Final Inference and Saving Results

A final parsing function is built to:
- Load longer MP3 recordings as tensors.
- Slice the audio into 3-second segments.
- Preprocess these segments into spectrograms.
- Run the model on these segments and group consecutive predictions.
- Count the number of capuchin calls per recording.

The model is saved to disk (`capuchin_best_model.h5`) and final predictions for each forest recording are saved in a CSV file (`Result.csv`).

---

## How to Run

1. **Prepare your data**: Place the audio files in the appropriate directories under the `data/` folder.
2. **Install Dependencies**: Ensure all required libraries are installed.
3. **Run the Main Script**: Execute the main Notebook:
   ```bash
   train cpu.ipynb
   ```
   This script will load the data, train the CNN model, display training metrics, perform predictions on test samples, and generate final results.
4. **Review Outputs**:
   - Model file: `Output/Model/capuchin_best_model.h5`
   - Prediction results: `Output/Result/Result.csv`
   - Training and evaluation plots will be displayed during execution.

---

## Results

After training the CNN model, key results include:
- High recall and precision values on both training and validation datasets.
- Visual plots indicating convergence of loss and accuracy metrics.
- Final inference on forest recordings showing the predicted number of capuchin calls per recording (for example, one recording with the highest count is `recording_08.mp3` with 25 calls).

These results demonstrate that the model successfully differentiates capuchin bird calls from other audio and is effective for large-scale inference.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Thanks to the HP Unlocked Challenge for providing the inspiration and dataset.
- Special thanks to the developers and contributors of TensorFlow, TensorFlow I/O, NumPy, and Matplotlib for their invaluable tools.
- Additional acknowledgment to any collaborators or mentors who provided feedback during development.