import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import kurtosis
from scipy import stats

# Denoising functions for salt-and-pepper noise
def remove_salt_and_pepper_noise_median(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Using Bilateral filter for salt-and-pepper noise
def remove_salt_and_pepper_noise_bilateral(image, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

# Using Gaussian filter for Gaussian noise
def remove_gaussian_noise_gaussian(image, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Denoising functions for Gaussian noise
def remove_gaussian_noise_nlm(image, d=9, sigmaColor=75, sigmaSpace=35):
    return cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)

# Functions to detect noise
def detect_salt_and_pepper_noise(image, window_size=(8, 8)):
    windowsize_r, windowsize_c = window_size
    noise_detected = []

    for r in range(0, image.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, image.shape[1] - windowsize_c, windowsize_c):
            window = image[r:r + windowsize_r, c:c + windowsize_c]
            kurt = kurtosis(window, axis=None)

            # Check for extreme pixel values indicative of salt and pepper noise
            if kurt > 3 and kurt < 5:  # Kurtosis threshold
                if np.min(window) == 0 and np.max(window) == 255:
                    noise_detected.append((r, c, window))

    return 1 if noise_detected else 0

def compute_variance_map(image, window_size=5):
    h, w = image.shape
    variance_map = np.zeros((h, w))
    padded_image = np.pad(image, window_size // 2, mode='reflect')

    for i in range(h):
        for j in range(w):
            window = padded_image[i:i + window_size, j:j + window_size]
            variance_map[i, j] = np.var(window)

    return variance_map

def detect_gaussian_noise(image):
    variance_map = compute_variance_map(image)
    flattened_variance = variance_map.flatten()
    stat, p_value = stats.shapiro(flattened_variance)

    if stat < 0.75:
        return 1
    else:
        return 0

# Metrics for comparison
def mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_value))

def ssim_metric(imageA, imageB):
    return ssim(imageA, imageB, data_range=imageB.max() - imageB.min())

# Function to rank filters and determine the best one
def determine_best_filter(mse1, psnr1, ssim1, mse2, psnr2, ssim2, filter_name1, filter_name2):
    score1 = 0
    score2 = 0

    # MSE comparison (lower is better)
    if mse1 < mse2:
        score1 += 1
    else:
        score2 += 1

    # PSNR comparison (higher is better)
    if psnr1 > psnr2:
        score1 += 1
    else:
        score2 += 1

    # SSIM comparison (higher is better)
    if ssim1 > ssim2:
        score1 += 1
    else:
        score2 += 1

    # Determine the best filter based on the score
    if score1 > score2:
        best_filter = filter_name1
    else:
        best_filter = filter_name2

    return best_filter

# Function to display performance metrics for the noisy image
def display_noisy_metrics(source_image, noisy_image):
    mse_value = mse(source_image, noisy_image)
    psnr_value = psnr(source_image, noisy_image)
    ssim_value = ssim_metric(source_image, noisy_image)

    print("Metrics for Noisy Image:")
    print(f"  MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")
    print("=" * 50)

# Function to process and compare images
def compare_denoising_results(original_image, denoised_image1, denoised_image2, image_path, output_folder, source_folder, filter_name1, filter_name2):
    source_filename = os.path.basename(image_path).replace("noisy_", "")
    source_image_path = os.path.join(source_folder, source_filename)
    source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)

    if source_image is not None:
        display_noisy_metrics(source_image, original_image)  # Display metrics for noisy image

        mse1 = mse(source_image, denoised_image1)
        psnr1 = psnr(source_image, denoised_image1)
        ssim1 = ssim_metric(source_image, denoised_image1)

        mse2 = mse(source_image, denoised_image2)
        psnr2 = psnr(source_image, denoised_image2)
        ssim2 = ssim_metric(source_image, denoised_image2)

        print(f"Metrics for {os.path.basename(image_path)}:")
        print(f"  {filter_name1}: MSE: {mse1:.2f}, PSNR: {psnr1:.2f}, SSIM: {ssim1:.4f}")
        print(f"  {filter_name2}: MSE: {mse2:.2f}, PSNR: {psnr2:.2f}, SSIM: {ssim2:.4f}")
        print("-" * 50)

        best_filter = determine_best_filter(mse1, psnr1, ssim1, mse2, psnr2, ssim2, filter_name1, filter_name2)
        print(f"Best filter for {os.path.basename(image_path)}: {best_filter}")
        print("=" * 50)

        output_path1 = os.path.join(output_folder, f"denoised_{filter_name1}_{os.path.basename(image_path)}")
        output_path2 = os.path.join(output_folder, f"denoised_{filter_name2}_{os.path.basename(image_path)}")
        cv2.imwrite(output_path1, denoised_image1)
        cv2.imwrite(output_path2, denoised_image2)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Noisy Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(denoised_image1, cmap='gray')
        plt.title(f'Denoised Image ({filter_name1})')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(denoised_image2, cmap='gray')
        plt.title(f'Denoised Image ({filter_name2})')
        plt.axis('off')

        plt.show()

# Processing a single image with comparisons
def process_single_image(image_path, output_folder, sharpen_folder, source_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        noisy_image = image.copy()

        if detect_gaussian_noise(image):
            print(f"Gaussian noise detected in {os.path.basename(image_path)}.")
            denoised_image_nlm = remove_gaussian_noise_nlm(image)
            denoised_image_gaussian = remove_gaussian_noise_gaussian(image)  # Using Gaussian filter
            compare_denoising_results(image, denoised_image_nlm, denoised_image_gaussian, image_path, output_folder, source_folder, "NLM", "Gaussian")

        elif detect_salt_and_pepper_noise(image):
            print(f"Salt-and-pepper noise detected in {os.path.basename(image_path)}.")
            denoised_image_median = remove_salt_and_pepper_noise_median(image)
            denoised_image_bilateral = remove_salt_and_pepper_noise_bilateral(image)
            compare_denoising_results(image, denoised_image_median, denoised_image_bilateral, image_path, output_folder, source_folder, "Median", "Bilateral")

# Function to process random images
def process_random_images(input_folder_salt_pepper, input_folder_gaussian, output_folder, sharpen_folder, source_folder, num_samples=5):
    salt_pepper_images = random.sample([f for f in os.listdir(input_folder_salt_pepper) if f.endswith(".jpg") or f.endswith(".png")], num_samples)
    gaussian_images = random.sample([f for f in os.listdir(input_folder_gaussian) if f.endswith(".jpg") or f.endswith(".png")], num_samples)

    print("Processing Salt and Pepper Noise Images...")
    for filename in salt_pepper_images:
        image_path = os.path.join(input_folder_salt_pepper, filename)
        process_single_image(image_path, output_folder, sharpen_folder, source_folder)

    print("\nProcessing Gaussian Noise Images...")
    for filename in gaussian_images:
        image_path = os.path.join(input_folder_gaussian, filename)
        process_single_image(image_path, output_folder, sharpen_folder, source_folder)

# Example Usage:
process_random_images('/content/drive/MyDrive/dataset',
                      '/content/drive/MyDrive/noisy_images',
                      '/content/drive/MyDrive/output1',
                      '/content/drive/MyDrive/sharpen1',
                      '/content/drive/MyDrive/source_grey',
                      num_samples=5)