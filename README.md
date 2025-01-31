# Lung Cancer Detection and Staging Using Deep Learning

## Overview
This project focuses on **lung cancer detection and staging** using **CT scan images** from the LIDC-IDRI dataset. A **U-Net model with a ResNet34 backbone** is used for **tumor segmentation**, and based on the segmented tumor size, the cancer is staged into different categories.

## Project Structure
```
capstone/
│── datasets/
│   ├── LIDC-IDRI-slices/
│   │   ├── LIDC-IDRI-XXXX/
│   │   │   ├── nodule-0/
│   │   │   │   ├── images/
│   │   │   │   ├── mask-0/
│   │   │   │   ├── mask-1/
│   │   │   │   ├── mask-2/
│   │   │   │   ├── mask-3/
│── research_papers/
│── script.ipynb
│── model.pth  # Saved model
```

## Dataset
- **Source:** LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)
- **Structure:** Each patient folder (`LIDC-IDRI-XXXX/`) contains:
  - `nodule-0/images/` - CT slices of the lung
  - `nodule-0/mask-{0,1,2,3}/` - Segmentation masks from 4 annotators

## Model Architecture
- **Pretrained U-Net with ResNet34 Backbone**
- **Input:** 128x128 grayscale CT scan images
- **Output:** Segmented lung tumor mask
- **Loss Function:** Dice Loss
- **Optimizer:** Adam (Learning rate = 1e-4)

## Training
1. **Data Preprocessing**:
   - Convert images to grayscale
   - Resize to 128x128
   - Normalize between [-1, 1]
2. **Mask Processing**:
   - Use all four available masks
   - Compute the **average mask** for training
3. **Training Setup**:
   - 80% of the dataset for training, 20% for testing
   - 10 epochs with batch size 16

## Performance Metrics
| Metric          | Score  |
|----------------|--------|
| Dice Score     | 0.7333 |
| IoU (Jaccard)  | 0.5843 |
| Accuracy       | 0.9978 |

## Cancer Staging Criteria
| Stage | Tumor Probability |
|-------|------------------|
| 0     | < 0.2            |
| 1     | 0.2 - 0.5        |
| 2     | 0.5 - 0.8        |
| 3     | > 0.8            |

## Prediction Function
- **Input:** CT scan image
- **Process:**
  - Pass through the model to get segmentation
  - Compute **tumor size** from the predicted mask
  - Assign a cancer stage based on probability thresholds
- **Output:**
  - Tumor Probability
  - Tumor Size (in pixels)
  - Predicted Cancer Stage

## Example Prediction
```
Predicted Tumor Probability: 0.67
Predicted Tumor Size (in pixels): 2450
Predicted Cancer Stage: 2 (Medium Tumor)
```

## Next Steps
- **Convert pixel size to real-world dimensions (mm/cm)**
- **Improve accuracy using multi-slice input aggregation**
- **Fine-tune model for clinical usage**

## How to Use
1. Train the model using `script.ipynb`
2. Save the model (`model.pth`)
3. Run predictions using:
```python
predicted_stage, tumor_size = predict_cancer_stage(image_path, model, device)
```

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- segmentation_models_pytorch
- numpy
- PIL

## Author
Aditya and team (Capstone Project)

