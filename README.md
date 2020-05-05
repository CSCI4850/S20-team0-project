# S20-team0-project

# Leabra Team
# Tumor Segmentation for Brain MRI
#
# DEMO
#
# Project Motivation:
Neural networks are mathematical models (often implemented as computer programs) that are  commonly used to make predictions on data, after having been trained on past data and outcomes. Using GPU acceleration is a common method to speed up training neural networks.

Medical imaging (MRI, X-ray, etc.) is useful in many medical applications. However, radiologist interpretations of medical images are not perfect. Computer-aided diagnosis (CAD) is an approach that uses software to analyze medical images as a second pair of eyes for a radiologist.

Neural networks can be used for CAD and can be trained to make predictions about medical images. U-net is a very good neural network for medical image segmentation, which shows where in an image a feature of interest exists. An example of medical image segmentation is predicting which pixels in a medical image depict a tumor.  

Training U-net on medical images benefits greatly from GPU acceleration. Having a limited setup in terms of GPUs can lead to running out of VRAM while training the traditional U-net.  A possible solution is to train a smaller version of U-net.

# What We Did / Main Aim:
We wanted to test how reducing the size of U-net would affect performance on tumor segmentation in brain MRI. We reduced the number of filters in the convolutional layers in U-net, which reduces the amount of VRAM required. 4 experiments were performed showing how U-net performance varies when it has all the filters, half the filters, a quarter of the filters, and an eighth of the filters. 

### Dependencies:
Install python 3.7.6 https://www.python.org/downloads/ 

Navigate to cloned repo and run this command in the terminal:
pip install -r requirements.txt


### How to Run a Jupyter Notebook:
https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html


# Demo: Option A    (Fast ~10 mins)
# Read the Demo 

### 1. Read the Demo
To see the performance of our trained models check out the demo here:
demo.ipynb

# Demo: Option B    (Slow ~3 days)
# Download Data, Normalize Data, and use Pretrained Models

### 1. Download the Data
NOTE* The dataset is ONLY available by request--a request may take several days to process. Follow the instructions here to request the data: https://www.med.upenn.edu/cbica/brats2019/registration.html 

Before running the demo, you will need to download the data, place it at the same level in the file structure as the cloned repo

### 2. Normalize the Data
Run the jupyter notebook named normalize_and_save_all_data.ipynb to save the normalized data.

This should produce a file structure like:

#### +-- MICCAI_BraTS_2019_Data_Training     //this is the dataset
#### |    +-- MICCAI_BraTS_2019_Data_Training
#### |    |     +-- HGG
#### |    |     +-- …
#### |    |     +-- normalized_hgg  //this folder contains the normalized data we just created
#### |    |     +-- ...
#### |
#### |
#### +-- S20-team0-project
#### |    +-- … //this is the cloned repo contents

### 3. Run the Demo using Pretrained Weights
Once the previous steps are accomplished, open demo.ipynb and run all cells in the notebook.

# Demo: Option C     (Slowest ~5 days) 
# Download Data, Normalize Data, and Train the Models Yourself


### 1. Download Data and Normalize Data
Follow steps 1 & 2 in Option B.

### 2. Train the models  (this overwrites the saved pretrained models)

There are 4 experiments to run.
NOTE* each experiment may take 10+ hours to run.

Run all the cells in the jupyter notebook called experiment_ds_1.ipynb
Run all the cells in the jupyter notebook called experiment_ds_2.ipynb
Run all the cells in the jupyter notebook called experiment_ds_4.ipynb
Run all the cells in the jupyter notebook called experiment_ds_8.ipynb

ds stands for the factor by which we are dividing the number of filters in each convolutional layer in U-net. Thus, ds_1 is the full U-net, ds_2 has half the filters, ds_4 has a quarter of the filters, and ds_8 has an eighth of the filters.

### 3. Run the Demo
Once the previous steps are accomplished, open demo.ipynb and run all cells in the notebook.
