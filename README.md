# Visual Question Answering for Flood Detection (VQA-FloodNet) Using YOLOv8, DeepLabV3+ and BERT 🌊💡

## 🌟 **Introduction**
FloodVQA is an innovative Visual Question Answering (VQA) system specifically designed to assist in flood disaster management. By leveraging the latest advancements in computer vision and natural language processing (NLP), the system provides intelligent answers to natural language questions about flood scenarios captured by UAVs (Unmanned Aerial Vehicles). This project aims to enhance disaster response and recovery by providing actionable insights from aerial imagery.

## 🛠️ **Overview**
This repository contains code for a **Visual Question Answering (VQA)** system designed for flood detection using a combination of **YOLOv8** (for object detection), **DeepLabV3+** (for semantic segmentation), and **BERT** (for question understanding). The project leverages pretrained models for feature extraction and incorporates a custom architecture for answering flood-related questions.

## 💡 **Project Objective**
The primary objective of this project is to:
- **Understand Flood Scenarios:** Identify objects and conditions in flood-affected areas.
- **Answer Questions:** Provide answers to structured and natural language queries about flood scenes.
- **Integrate Multi-Modal Features:** Fuse data from object detection, semantic segmentation, and NLP for accurate predictions.


## 🎨 **Installation and Dependencies**
### **Prerequisites**
- **Python 3.8+**
- **Google Colab** or a local environment with GPU support
- Required **Python libraries** (installed via pip):
  - `torch`, `torchvision`
  - `transformers`
  - `ultralytics`
  - `segmentation-models-pytorch`
  - `tqdm`, `Pillow`
Install the required libraries using the following commands:
```bash
pip install ultralytics transformers segmentation-models-pytorch tqdm pillow
pip install torch torchvision
```
### **Mount Google Drive (if using Colab)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### **Ensure GPU support**
   - The code will automatically detect GPU if available, else fallback to CPU.
   - You can check CUDA availability:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```
### **Clone Repository**
```bash
git clone https://github.com/AbdelkadirSellahi/VQA-Using-YOLO-DeepLabV3-BERT.git
cd FloodVQA
```



## 📂 **Dataset**
The dataset used is from the [**FloodNet Challenge (EarthVision 2021 - Track 2)**](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021?fbclid=IwAR2XIwe5nJg5VSgxgCldM7K0HPtVsDxB0fjd8cJJZfz6WMe3g0Pxg2W3PlE). It consists of:
- **Images:** UAV-captured images of flood scenes.
- **Questions:** Structured natural language queries about the scenes.
- **Answers:** Ground-truth labels for the queries.

### Dataset Structure:
- **Images Directory:** Contains raw images for training and testing.
- **Annotations File (JSON):** Maps image IDs to:
  - Questions.
  - Ground truth answers.
  - Question types (e.g., Yes/No, Counting, Condition Recognition).

### Data Processing:
- Images are resized to **224x224**.
- Questions are tokenized using **BERTTokenizer**.
- Labels (answers) are encoded using a predefined **label mapping**.

## 🧪 **Models Used**
### **a. Object Detection: YOLOv8**
- **Purpose:** Detect objects like buildings, roads, and trees in flood scenes.
- **Features Extracted:** Bounding boxes and class predictions.
- **Pretrained Weights:** Fine-tuned on flood-related data.

### **b. Semantic Segmentation: DeepLabV3+**
- **Architecture:** DeepLabV3+ with a ResNet50 backbone.
- **Task:** Segment flood-related objects in images.
- **Input Size:** Images resized to **256x256**.
- **Features Extracted:** Compressed feature maps using adaptive pooling.

### **c. Natural Language Processing: BERT**
- **Model Used:** BERT-base-uncased.
- **Task:** Process and understand natural language questions.
- **Tokenization:** Questions converted into numerical tensors.

### **d. VQA Model**
- Combines:
  - YOLO features (bounding boxes + classes).
  - DeepLab features (semantic maps).
  - Text features from BERT.
- Outputs an answer based on the fused features.

## 🧪 Usage

### 1. Prepare the Models
- Ensure pretrained models are available:
  - YOLOv8 weights (`best.pt`): Place in `/content/drive/MyDrive/.../weights/`.
  - DeepLabV3+ weights: Specify the checkpoint path in the script.
  - BERT model: Loaded directly from Hugging Face.

### 2. Load the Models
```python
from ultralytics import YOLO
yolo_model = YOLO('/path/to/yolo/best.pt').to(device)
deeplab_model, _, _, _ = init_model(args)
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
```


## 🔧 **Hyperparameters**
### **Training Parameters**
- **Batch Size:** 128 for VQA training.
- **Learning Rate:** 
  - 0.00001 for the VQA model.
- **Optimizer:** Adam.
- **Loss Function:** CrossEntropyLoss.
- **Number of Epochs:**
  - 10 epochs for VQA model training.

### **Model-Specific Parameters**
- **YOLOv8 Input Dimension:** 5 (bounding boxes and class labels).
- **DeepLab Output Dimension:** Compressed to **7x7x10**.
- **BERT Combined Dimension:** 768.
- **Hidden Dimension:** 256 for intermediate layers.

## 🎨 **Model Architecture**
### **a. YOLOv8 for Object Detection**
- **Role:** Detect flood-relevant objects and generate bounding box features.
- **Inputs:** Resized images.
- **Outputs:** A tensor of bounding boxes and associated class predictions.

### **b. DeepLabV3+ for Semantic Segmentation**
- **Architecture:** ResNet50 backbone with atrous spatial pyramid pooling (ASPP).
- **Role:** Segment objects like buildings, roads, and trees.
- **Inputs:** Preprocessed images.
- **Outputs:** Semantic feature maps.

### **c. BERT for NLP**
- **Role:** Tokenize and extract contextual embeddings from questions.
- **Inputs:** Tokenized question text.
- **Outputs:** A 768-dimensional feature vector.

### **d. VQA Model Pipeline**
- **Feature Fusion:**
  - YOLO and DeepLab features are flattened and passed through a dense layer.
  - BERT embeddings are concatenated with vision features.
- **Classifier:**
  - Fused features passed through fully connected layers to predict answers.
  
## 🚀 **Code Overview**
### **a. Directory Structure**
- **/Codes/semantic segmentation pytorch:** DeepLabV3+ model.
- **/00_PFE/Object_Detection:** YOLOv8 model.
- **/00_PFE/VQA/Code-V6:** Code for VQA training and model checkpoints.
- **/00_PFE/DataSet:** FloodNet dataset with subfolders for images and annotations.

#### **Files:**
- **utils/train.py:** Contains helper functions for model initialization and training loops.
- **log.csv:** Logs training metrics (loss, accuracy) for VQA model.
- **VQAModel_Best.pth:** Saved state dictionary of the best VQA model.

### **b. Key Files**
1. **`train.py`**
   - Central training script for the VQA model.
   - Functions:
     - `initialize_models`: Loads YOLOv8, DeepLabV3+, and BERT components.
     - `train_one_epoch`: Executes a training loop for one epoch.
     - `evaluate_model`: Validates the model on a test set.
   - Output:
     - Saved checkpoints for the best-performing model.

2. **`utils/dataloader.py`**
   - Custom DataLoader for loading:
     - Image features.
     - Questions and tokenized embeddings.
     - Ground-truth answers.
   - Preprocessing:
     - Resizing images and generating feature tensors.

3. **`VQAModel.py`**
   - Defines the custom VQA model architecture:
     - Vision feature extractors.
     - Question encoder.
     - Fully connected layers for classification.

4. **`log.csv`**
   - Tracks:
     - Training loss and validation loss.
     - Accuracy metrics per epoch.
    




### **c. Training Pipeline**
1. **Feature Extraction**
   - YOLOv8 extracts bounding boxes and object class predictions.
   - DeepLabV3+ provides semantic maps from resized images.
   - BERT processes tokenized questions.
2. **Data Loading:**
   - Custom DataLoader (`VQADataset`) groups questions and answers by image.

3. **VQA Model Training**
   - Combine vision features with question embeddings.
   - Pass the fused features through fully connected layers.
   - Compute classification loss using `CrossEntropyLoss`.

4. **Metrics:**
   - Loss calculated using CrossEntropy.
   - Accuracy computed by comparing predictions to ground-truth labels.

5. **Checkpointing:**
   - Best-performing model saved to a file

6. **Validation**
   - Validate the model on a held-out test set.
   - Metrics include:
     - Accuracy for Yes/No and recognition tasks.
     - Mean Absolute Error (MAE) for counting tasks.

7. **Fine-Tuning**
   - Adjust learning rates and hidden dimensions to optimize performance.


## 💡 **Results**
The system achieves:
- **High Accuracy** on Yes/No and Condition Recognition questions.
- **Low MAE** on counting tasks.

### **Sample Question-Answer Outputs**
| **Image**                | **Question**                  | **Answer**           |
|---------------------------|-------------------------------|----------------------|
| DataSet/sample1.jpg | "How many buildings are flooded?" | 3                    |
| DataSet/sample2.jpg | "Is the road passable?"          | No                   |

## 🚀 **Challenges and Problematic**
- **Integration Complexity:** Combining object detection, segmentation, and NLP pipelines.
- **Training Time:** High computational requirements due to model size.
- **Feature Alignment:** Ensuring features from YOLO, DeepLab, and BERT align spatially and dimensionally.


## 💬 **Contributing**

We welcome contributions! If you have ideas or improvements:
1. Fork the repository.
2. Create a new branch: `git checkout -b my-feature-branch`.
3. Submit a pull request.


## 💬 **Contact**

Feel free to open an issue or reach out for collaboration!  

**Author**: *Abdelkadir Sellahi*

**Email**: *abdelkadirsellahi@gmail.com* 

**GitHub**: [Abdelkadir Sellahi](https://github.com/AbdelkadirSellahi)
