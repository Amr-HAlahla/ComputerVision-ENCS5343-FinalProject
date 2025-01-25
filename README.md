# Deep Learning Project: Comparing CNN Architectures for Classification Tasks

## Overview
This repository contains the implementation of a deep learning project that explores and evaluates different convolutional neural network (CNN) architectures on a classification dataset. The work is organized into four tasks, each focusing on a unique approach to achieve robust performance. The project uses **TensorFlow/Keras** for building and training the models and concludes with a detailed analysis of results.

## Contents
- **Task 1**: Designing and experimenting with custom CNN architectures.
- **Task 2**: Utilizing data augmentation techniques to train the best custom architecture.
- **Task 3**: Training a well-known CNN architecture (MobileNetV1) from scratch.
- **Task 4**: Applying transfer learning using MobileNetV1 pretrained on ImageNet.

---

## Repository Structure
```plaintext
- vision-FinalProject.ipynb    # Implementation of all tasks
- Project-2024.pdf   # Description file for the project
- CV-FinalProject-Report.pdf      # Final report for the project
- README.md                 # Repository documentation
```
---

## Methodology
### Task 1: Custom CNN Architectures
- Implemented and evaluated various CNN architectures, including:
  - 4 Convolutional + 2 Fully Connected layers.
  - 5 Convolutional + 2 Fully Connected layers.
  - Wider and deeper versions with dynamic filter sizes and pooling layers.
- Conducted experiments with different configurations of dropout, batch normalization, and fully connected layers.
- Selected the **Wider 5Conv + 2FC** architecture based on its superior performance.

### Task 2: Training with Data Augmentation
- Augmented the dataset using techniques such as flipping, rotation, and zoom.
- Trained the Wider 5Conv + 2FC architecture on the augmented dataset.
- Achieved a test accuracy of **92.65%** with a test loss of **0.33**.

### Task 3: Training MobileNetV1 from Scratch
- Trained the **MobileNetV1** architecture with grayscale images resized to 224x224.
- Obtained a test accuracy of **92.39%**, demonstrating competitive performance.
- Highlighted the resource-intensive nature of training MobileNetV1 from scratch.

### Task 4: Transfer Learning with MobileNetV1
- Applied transfer learning by fine-tuning MobileNetV1 pretrained on ImageNet.
- Achieved the highest test accuracy of **94.96%**, demonstrating the advantage of transfer learning for this dataset.
- Reduced training time significantly compared to training from scratch.

---

## Results and Insights
- **Task 1**: Provided insights into the impact of architecture depth, width, and dropout configurations.
- **Task 2**: Validated the effectiveness of custom architectures in achieving high accuracy.
- **Task 3**: Demonstrated the feasibility of training a well-known model from scratch, albeit with higher resource requirements.
- **Task 4**: Highlighted the significant benefits of leveraging pretrained models for transfer learning.

| Task | Architecture | Test Accuracy | Test Loss | Notes |
|------|--------------|---------------|-----------|-------|
| Task 1 | Wider 5Conv + 2FC | 90.79% | 0.31 | Best custom architecture. |
| Task 2 | Wider 5Conv + 2FC | 92.69% | 0.33 | Retrained on augmented data. |
| Task 3 | MobileNetV1 (Scratch) | 92.39% | 0.31 | Trained from scratch. |
| Task 4 | MobileNetV1 (Transfer Learning) | 94.96% | 0.24 | Leveraged pretrained weights. |

---

## Figures
### Accuracy and Loss Plots
Each task's training and validation accuracy/loss is visualized to showcase the model's learning behavior and generalization performance. These plots are included in the notebook file and demonstrate the progression of each approach.

---

## Prerequisites and How to Run

### Prerequisites
This project requires the following dependencies:
- Python 3.7+
- TensorFlow/Keras 2.x
- OpenCV
- NumPy
- Matplotlib

You can install the required libraries using the following command:
```bash
pip install tensorflow opencv-python numpy matplotlib
```
### How to Run
The notebook was developed and tested using Kaggle's resources, which provide free access to GPUs for efficient model training. Alternatively, you can use platforms like Google Colab for similar benefits if local device resources are insufficient. To run the notebook:

1. Clone the repository:
   ```bash
   git clone https://github.com/Amr-HAlahla/ComputerVision-ENCS5343-FinalProject.git
  ``
2. Navigate to the repository directory:
   ```bash
  cd ComputerVision-ENCS5343-FinalProject
  ```
3. Open the Jupyter Notebook::
   ```bash
  jupyter notebook project_notebook.ipynb
  ``
4. Run all cells to replicate the results and visualizations.

**Note**: For Kaggle or Google Colab, upload the notebook file to the respective platform, adjust the dataset paths if necessary, and run the cells to execute the implementation.

