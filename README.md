This repository contains an end-to-end machine learning pipeline for Speech Separation, Transcription (ASR), and Diarization. The project leverages deep learning architectures like Conv-TasNet for audio separation and advanced signal processing techniques for enhancement and evaluation.

Project Overview
The main goal of this project is to take a mixed audio recording of multiple speakers and perform the following:

Speaker Separation: Isolate individual voices from the mixture using a Conv-TasNet model.

Speech Enhancement: Refine the separated audio using Wiener Filtering to improve signal quality.

ASR & Diarization: Transcribe the isolated speech and generate a diarized output table in the format time | speaker | text.

Performance Evaluation: Measure the quality of separation using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric.

Repository Structure
Team13_final.ipynb: The primary pipeline notebook.

Environment setup and dependency installation.

Model training and transfer learning for speaker separation.

Inference workflow for ASR and diarization.

result_analysis.ipynb: The post-processing and evaluation notebook.

Calculation of baseline SI-SDR scores.

Implementation of Wiener filtering for enhancement.

Permutation alignment to match predictions with ground truth signals.

Visual analysis of improvement (Δ SI-SDR).

Technical Features
Model Architecture & Tools
Separation: Conv-TasNet (Convolutional Time-domain Audio Separation Network).

Transcription: faster-whisper for high-performance Automatic Speech Recognition.

Analysis: resemblyzer for speaker embedding and diarization tasks.

Framework: Built using PyTorch and Torchaudio.

Evaluation Metrics
The project focuses on SI-SDR as the core metric to evaluate separation accuracy. Because separation models may output speakers in a different order than the ground truth, the pipeline includes a permutation solver to find the optimal speaker mapping before scoring.

Setup & Installation
Dependencies
The project requires the following Python libraries:

torch, torchaudio

faster-whisper

resemblyzer

soundfile

matplotlib, pandas, numpy

Installation
You can install the necessary packages using pip:

Bash

pip install torch torchaudio numpy pandas tqdm matplotlib faster-whisper resemblyzer scikit-learn torchcodec soundfile
How to Use
Prepare Data: Upload your mixed audio (e.g., mixed.wav) and corresponding ground truth files (if available for evaluation) to Google Drive.

Run Separation: Use Team13_final.ipynb to process the mixed audio. This will generate individual .wav files for each detected speaker and a diarized transcript.

Enhance & Evaluate: Use result_analysis.ipynb to apply Wiener filtering to the outputs and generate a performance report with SI-SDR improvements.
