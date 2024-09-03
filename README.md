# Denoising Autoencoder for Medical Image Enhancement

## Overview

This project implements a denoising autoencoder to enhance medical images, such as X-rays, by removing noise and improving image clarity. The goal is to aid in more accurate diagnoses by providing clearer medical images.

## Dataset

The dataset used in this project is the [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) containing X-ray images. It includes images from patients diagnosed with COVID-19, as well as normal and viral pneumonia cases.


## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Shreyas_jadv/denoising-autoencoder-medical-image-enhancement.git
    cd denoising-autoencoder-medical-image-enhancement
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**

    Download the COVID-19 Radiography Database from [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) and place it in the project directory.

5. **Update paths:**

    In the `denoising_autoencoder.py` script, update the `train_dir` and `test_dir` variables to point to the correct dataset paths:

## Running the Project

To train the denoising autoencoder and visualize the results:

```bash
python denoising_autoencoder.py


