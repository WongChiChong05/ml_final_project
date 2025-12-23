<h2>
This repository contains an end-to-end machine learning pipeline for Speech Separation, Transcription (ASR), and Diarization. The project leverages deep learning architectures like Conv-TasNet for audio separation and advanced signal processing techniques for enhancement and evaluation. </h2>

<h2> Project Overview </h3>

<h3> The main goal of this project is to take a mixed audio recording of multiple speakers and perform the following: </h3><br>
<ul>
<li>Speaker Separation: Isolate individual voices from the mixture using a Conv-TasNet model.</li>

<li> Speech Enhancement: Refine the separated audio using Wiener Filtering to improve signal quality.</li>

<li> ASR & Diarization: Transcribe the isolated speech and generate a diarized output table in the format time | speaker | text.</li>

<li> Performance Evaluation: Measure the quality of separation using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric.</li>

</ul>

<h2> Repository Structure </h2>
<ul>
Team13_final.ipynb: The primary pipeline notebook.
<ul>
<li>Environment setup and dependency installation.</li>

<li>Model training and transfer learning for speaker separation.</li>

<li>Inference workflow for ASR and diarization.</li>
</ul>
</ul>

<ul>
result_analysis.ipynb: The post-processing and evaluation notebook.
<ul>
<li>Calculation of baseline SI-SDR scores.</li>

<li>Implementation of Wiener filtering for enhancement.</li>

<li>Permutation alignment to match predictions with ground truth signals.</li>

<li>Visual analysis of improvement (Δ SI-SDR).</li>
</ul>
</ul>

<h2>Dataset</h2>
The model is trained using a synthetic mixture approach derived from the LibriSpeech dataset.
<ul>
<li>Source Data: Includes train-clean-100, dev-clean, and test-clean subsets from OpenSLR.</li>

<li>Mixture Generation: Audio samples are mixed on-the-fly using a custom MixtureDataset class. It randomly selects 4 speakers, resamples the audio to 8,000 Hz, and clips or pads them to a 2.0-second duration to create a mixed signal.</li>

<li> Configuration: </li>
<ul>
<li> Speakers: 4-speaker mixtures.</li>

<li>Sample Rate: 8,000 Hz.</li>

<li>Duration: 2-second audio segments.</li>
</ul>
</ul>

<h2>Model Training</h2>
The core separation engine is a Conv-TasNet (Convolutional Time-domain Audio Separation Network).
<h3>Training Workflow</h3>
<ol>

<li>Architecture: The model consists of an Encoder (1D Convolution), a Separation Block (Temporal Convolutional Network), and a Decoder (Transposed 1D Convolution).
</li>
<li>
Loss Function: Employs Permutation Invariant Training (PIT) based on the SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) loss. This allows the model to correctly match predicted signals to ground truth even if the output order varies.
</li>
<li>
Optimization:
<ul>
<li>Learning Rate: 1e-4.</li>

<li>Batch Size: 4.</li>

<li>Epochs: 50.</li>
</ul></li>
<li>Hardware: Developed and trained using high-performance GPU environments (e.g., NVIDIA A100).</li>
</ol>
<h2> Technical Features </h2>
<h3>Model Architecture & Tools</h3>
<ul>
<li>Separation: Conv-TasNet (Convolutional Time-domain Audio Separation Network).</li>

<li>Transcription: faster-whisper for high-performance Automatic Speech Recognition.</li>

<li>Analysis: resemblyzer for speaker embedding and diarization tasks.</li>

<li>Framework: Built using PyTorch and Torchaudio.</li>
</ul>

<h3> Evaluation Metrics </h3>
The project focuses on SI-SDR as the core metric to evaluate separation accuracy. Because separation models may output speakers in a different order than the ground truth, the pipeline includes a permutation solver to find the optimal speaker mapping before scoring.

<h2> Setup & Installation </h2>
<h3>Dependencies</h3>

The project requires the following Python libraries:
<ul>
<li>torch, torchaudio</li>

<li>faster-whisper</li>

<li>resemblyzer</li>

<li>soundfile</li>

<li>matplotlib, pandas, numpy</li>
</ul>
<h2>Installation</h2>
<h3>The project is optimized for Google Colab but can be run locally with a GPU.</h3>
<h3>Dependencies</h3>
```bash
pip install torch torchaudio numpy pandas tqdm matplotlib faster-whisper resemblyzer scikit-learn soundfile
```
<h3>Usage Instructions</h3>
<ol>
<li>Run Team13_final.ipynb: This will auto-download the LibriSpeech dataset, and train the model. After training, you can provide a mixed audio file to generate the separated tracks and transcript.</li>

<li>Run result_analysis.ipynb: Use this to compare your predicted audio against ground truth signals. It will generate a report showing the SI-SDR Improvement (Δ) achieved by the Wiener Filter.</li>
</ol>

<h2>How to Use</h2>
<ol>
<li>Prepare Data: Upload your mixed audio (e.g., mixed.wav) and corresponding ground truth files (if available for evaluation) to Google Drive.</li>

<li>Run Separation: Use Team13_final.ipynb to process the mixed audio. This will generate individual .wav files for each detected speaker and a diarized transcript.</li>

<li>Enhance & Evaluate: Use result_analysis.ipynb to apply Wiener filtering to the outputs and generate a performance report with SI-SDR improvements.</li>
</ol>
