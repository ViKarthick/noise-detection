# Advanced Noise Detection and Denoising in Grayscale Images

## Overview
This repository contains a Python-based project for detecting and denoising noise (salt-and-pepper and Gaussian) in grayscale images. Developed as part of my undergraduate research at Sri Sivasubramaniya Nadar College of Engineering (Affiliated to Anna University), this project showcases my skills in image processing, statistical analysis, and AI, aligning with my pursuit of a Master’s in AI for Fall 2026.

## Features
- **Noise Detection**: 
  - Salt-and-pepper noise detected using kurtosis (threshold 3–5) on local image windows.
  - Gaussian noise identified via variance mapping and Shapiro-Wilk statistical testing.
- **Denoising Techniques**: 
  - Salt-and-pepper: Median and Bilateral filters.
  - Gaussian: Gaussian Blur and Non-Local Means (NLM) denoising.
- **Performance Metrics**: Evaluates Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) to compare denoising methods.
- **Best Filter Selection**: Automatically determines the optimal filter based on MSE, PSNR, and SSIM scores.
- **Visualization**: Displays original and denoised images using Matplotlib for easy comparison.

## Innovation
- **Kurtosis-Based Detection**: Utilizes kurtosis to segment salt-and-pepper noise, a creative adaptation from signal processing, tailored for image analysis.
- **Statistical Noise Classification**: Employs the Shapiro-Wilk test on variance maps to detect Gaussian noise, blending probability and image processing in a novel way for an undergraduate project.
- **Multi-Metric Optimization**: Introduces a data-driven approach to select the best denoising filter, enhancing traditional methods.

## Results
- Sample outputs demonstrate effective noise removal, with bilateral and NLM filters outperforming in specific cases.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vikarthick24122004/noise-detection.git
   cd noise-detection
