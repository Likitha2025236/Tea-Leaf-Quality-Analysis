#**Tea Leaf Quality Analysis using Advanced Deep Transfer Learning Method**


**Description**
This project presents a deep transfer learning approach to predict the quality of Longjing tea leaves using Hyperspectral Imaging (HSI) data. The proposed model integrates Convolutional Neural Networks (CNN), Capsule Networks, and an Improved Pooling Attention (IPA) mechanism to extract and analyze spatial and spectral features. The project includes data augmentation, transfer learning strategies, and hyperparameter tuning to improve model generalizability and accuracy, even with limited labeled data.

**Dataset Information**

•	Dataset: Longjing Tea Leaf Dataset

•	Classes: C1, C2, C3, C4, C5, C6 (representing tea grades)

•	Modality: Hyperspectral Images

•	Characteristics: Low spatial resolution, high spectral resolution

•	Size: Limited number of labeled samples (augmented for training)

Code Information
The codebase contains the following main components:
•	data_preprocessing.py: Handles image preprocessing, augmentation, and splitting into train/test/validation sets.
•	model.py: Contains the architecture combining CNN, Capsule Network, and Improved Pooling Attention (IPA).
•	train.py: Trains the model using the selected hyperparameters.
•	evaluate.py: Computes accuracy, precision, recall, F1-score.
•	utils/: Includes helper functions for loading data, visualization, and metrics.
•	experiments/: Contains scripts and logs of hyperparameter tuning results.

Usage Instructions 
Step 1: Open the Project
1.	Open Google Colab in your browser.
2.	Create a new notebook or upload the provided .ipynb file.
Step 2: Set Up the Runtime
1.	Click on Runtime in the top menu.
2.	Select Change runtime type.
3.	Set Hardware Accelerator to GPU.
4.	Click Save.
Step 3: Upload or Mount Dataset
Option A: Upload Manually
1.	On the left pane, click the folder icon.
2.	Click the upload icon and select your dataset files.
Step 4: Install the Libraries 
Run the following code in a cell to install any missing packages:
!pip install numpy pandas matplotlib scikit-learn torch torchvision
Step 5: Load the Dataset
Modify the code to load the dataset correctly. 

Step 6: Run the Code
1.	Copy and paste the full model training code into a Colab cell.
2.	Press Shift + Enter to run the cell.
Step 7: Train the Model
The model will automatically begin training. 
Step 8: Evaluate & Save the Model

Requirements
To successfully run the code for Tea Leaf Quality Analysis using Advanced Deep Transfer Learning, ensure the following dependencies are installed:
Python Version
•	Python 3.7 or later
Required Libraries
Library	Version	Purpose
numpy	≥ 1.19	Numerical operations
pandas	≥ 1.1	Data handling and preprocessing
matplotlib	≥ 3.2	Visualization
scikit-learn	≥ 0.24	Metrics and data splitting
torch	≥ 1.9	Deep learning framework (PyTorch)
torchvision	≥ 0.10	Useful utilities for computer vision tasks
tqdm	≥ 4.62	Progress bar for training visualization

Installation Command
Install all dependencies using the following command:
pip install numpy pandas matplotlib scikit-learn torch torchvision tqdm

Methodology – Steps for Data Processing and Modeling
The following steps outline the methodology used in Tea Leaf Quality Analysis using Advanced Deep Transfer Learning:

1 Data Collection- Hyperspectral Imaging (HSI) data of Longjing tea leaves was acquired. Dataset includes images categorized into six quality grades: C1 to C6.

2. Data Preprocessing
•	Noise removal: Applied denoising techniques to handle noisy HSI bands.
•	Normalization: Spectral data was normalized to a fixed scale.
•	Augmentation: Data augmentation (rotation, flipping, etc.) was performed to increase sample size and variability.
•	Dimensionality reduction: Characteristic bands were selected to reduce redundancy using statistical and spectral techniques.

3. Model Architecture
•Combined CNN, Capsule Networks, and Improved Pooling Attention (IPA) mechanism for deep feature extraction.
•	CNN extracts low-level spatial features.
•	Capsule Network captures spatial-spectral relationships.
•	IPA module improves discriminative power by dynamically pooling relevant information.

4. Transfer Learning
•	Pretrained on an augmented HSI dataset.
•	Transferred knowledge was fine-tuned on the Longjing tea dataset using a meta-baseline framework for few-shot learning.

5. Training Configuration
•	Split: 70% training, 20% validation, 10% testing.
•	Optimized using:
o	Epochs: 100
o	Batch size: 4
o	Best learning rate: 0.0001
•	Loss function: Cross-entropy
•	Optimizer: Adam

6. Performance Evaluation
•	Used metrics: Accuracy, Precision, Recall, F1-score
•	Compared against baseline models: CNN, LSTM, CNN-LSTM, and CNN-Capsule.

7. Ablation Study
•	Compared different pooling methods:
o	Average Pooling (AP)
o	Max Pooling (MP)
o	Improved Pooling Attention (IPA) — demonstrated highest performance
