# Reproducibility Package – SemEval-2026 Task 2 (Subtask 2A)

This repository contains the training and inference code for our system for SemEval-2026 Task 2 – Subtask 2A (State Change Forecasting).

## 1. System description

### Data used for training

We use only the official SemEval-2026 Task-2 Subtask-2A training dataset.  
Each instance contains a user identifier, a text entry, previous self-reported affect values, and the corresponding state change labels for valence and arousal.  
No additional external datasets or resources are used.

### Model inputs and outputs

The input to the system consists of the last observed text for a user along with the corresponding previous valence and arousal values.  
The model outputs two real-valued continuous predictions:

- predicted state change in valence  
- predicted state change in arousal  

Both values represent signed changes and can be positive or negative, following the task definition.

### Model and training procedure

The system is based on a pretrained RoBERTa encoder followed by a bidirectional LSTM layer and a multilayer perceptron regression head.  
Previous valence and arousal values, user embeddings, and lightweight surface features are incorporated to personalize predictions.  
The model is trained using mean squared error (MSE) loss over both outputs jointly.  
Lower layers of the RoBERTa encoder are frozen and only the upper transformer layers and regression components are fine-tuned.  
User-based cross-validation is used during development, and a final model is trained on the full training set.

### Example

Training example  

Input:  
Text: I feel exhausted after today’s shift.  
Previous affect: valence = 1.0, arousal = 1.5  

Target:  
state change valence = -0.8, state change arousal = -0.6  

Inference example  

Input:  
Text: I am feeling overwhelmed and tired.  
Previous affect: valence = 1.0, arousal = 1.5  

Prediction:  
state change valence = -0.72, state change arousal = -0.55  

## 2. Code and model artifacts

The package contains:

- train_2a.py – training script  
- infer_2a.py – inference script  
- model.py – model architecture definition  
- models/final_subtask2a_model.pt – trained model checkpoint  

## Model weights
Download the trained model from:
https://drive.google.com/file/d/1eI-vg0JZ4UFtlwQzsPXFGBNgpgwr5QKO/view?usp=sharing

Place the file at:
models/final_subtask2a_model.pt

## 3. How to run the system

### Environment setup

pip install -r requirements.txt

### Training

python train_2a.py --train_file path/to/train_subtask2a.csv --output_model models/final_subtask2a_model.pt

### Inference

python infer_2a.py --model_path models/final_subtask2a_model.pt --test_file path/to/subtask2a_forecasting_user_marker.csv --output_file pred_subtask2a.csv

Use subtask2a_forecasting_user_marker.csv for inference.

Predict only users with is_forecasting_user == True (one row per user).

The output file follows the format:

user_id,pred_state_change_valence,pred_state_change_arousal




