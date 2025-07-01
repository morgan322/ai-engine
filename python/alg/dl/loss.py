import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set font for better compatibility (supports English)
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display

# 1. L1 Loss (Mean Absolute Error)
def l1_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# 2. MSE Loss (Mean Squared Error)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 3. Cross-Entropy Loss (CE)
def ce_loss(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 4. KL Divergence
def kl_divergence(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1)
    y_true = np.clip(y_true, epsilon, 1)
    return np.mean(y_true * np.log(y_true / y_pred))

# 5. Binary Cross-Entropy (BCE)
def bce_loss(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 6. BCEWithLogitsLoss
def bce_with_logits_loss(y_true, logits):
    y_pred = 1 / (1 + np.exp(-logits))  # Sigmoid
    return bce_loss(y_true, y_pred)

# 7. Focal Loss
def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    return -np.mean(alpha * (1 - pt) ** gamma * np.log(pt))

# 8. IoU Loss
def iou_loss(boxes_true, boxes_pred):
    boxes_true = np.array(boxes_true)
    boxes_pred = np.array(boxes_pred)
    
    x1 = np.maximum(boxes_true[:, 0], boxes_pred[:, 0])
    y1 = np.maximum(boxes_true[:, 1], boxes_pred[:, 1])
    x2 = np.minimum(boxes_true[:, 2], boxes_pred[:, 2])
    y2 = np.minimum(boxes_true[:, 3], boxes_pred[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_true = (boxes_true[:, 2] - boxes_true[:, 0]) * (boxes_true[:, 3] - boxes_true[:, 1])
    area_pred = (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])
    union = area_true + area_pred - intersection
    
    iou = intersection / (union + 1e-10)
    return 1 - np.mean(iou)

# 9. GIoU Loss
def giou_loss(boxes_true, boxes_pred):
    boxes_true = np.array(boxes_true)
    boxes_pred = np.array(boxes_pred)
    
    # Calculate intersection
    x1_i = np.maximum(boxes_true[:, 0], boxes_pred[:, 0])
    y1_i = np.maximum(boxes_true[:, 1], boxes_pred[:, 1])
    x2_i = np.minimum(boxes_true[:, 2], boxes_pred[:, 2])
    y2_i = np.minimum(boxes_true[:, 3], boxes_pred[:, 3])
    intersection = np.maximum(0, x2_i - x1_i) * np.maximum(0, y2_i - y1_i)
    
    # Calculate union
    area_true = (boxes_true[:, 2] - boxes_true[:, 0]) * (boxes_true[:, 3] - boxes_true[:, 1])
    area_pred = (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])
    union = area_true + area_pred - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-10)
    
    # Calculate smallest enclosing box
    x1_c = np.minimum(boxes_true[:, 0], boxes_pred[:, 0])
    y1_c = np.minimum(boxes_true[:, 1], boxes_pred[:, 1])
    x2_c = np.maximum(boxes_true[:, 2], boxes_pred[:, 2])
    y2_c = np.maximum(boxes_true[:, 3], boxes_pred[:, 3])
    area_c = (x2_c - x1_c) * (y2_c - y1_c)
    
    # Calculate GIoU
    giou = iou - (area_c - union) / (area_c + 1e-10)
    return 1 - np.mean(giou)

# 10. DIoU Loss
def diou_loss(boxes_true, boxes_pred):
    boxes_true = np.array(boxes_true)
    boxes_pred = np.array(boxes_pred)
    
    # Calculate intersection
    x1_i = np.maximum(boxes_true[:, 0], boxes_pred[:, 0])
    y1_i = np.maximum(boxes_true[:, 1], boxes_pred[:, 1])
    x2_i = np.minimum(boxes_true[:, 2], boxes_pred[:, 2])
    y2_i = np.minimum(boxes_true[:, 3], boxes_pred[:, 3])
    intersection = np.maximum(0, x2_i - x1_i) * np.maximum(0, y2_i - y1_i)
    
    # Calculate union
    area_true = (boxes_true[:, 2] - boxes_true[:, 0]) * (boxes_true[:, 3] - boxes_true[:, 1])
    area_pred = (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])
    union = area_true + area_pred - intersection
    iou = intersection / (union + 1e-10)
    
    # Calculate center distance
    ctr_true_x = (boxes_true[:, 0] + boxes_true[:, 2]) / 2
    ctr_true_y = (boxes_true[:, 1] + boxes_true[:, 3]) / 2
    ctr_pred_x = (boxes_pred[:, 0] + boxes_pred[:, 2]) / 2
    ctr_pred_y = (boxes_pred[:, 1] + boxes_pred[:, 3]) / 2
    
    dist = np.sqrt((ctr_true_x - ctr_pred_x) **2 + (ctr_true_y - ctr_pred_y)** 2)
    
    # Calculate diagonal length of smallest enclosing box
    x1_c = np.minimum(boxes_true[:, 0], boxes_pred[:, 0])
    y1_c = np.minimum(boxes_true[:, 1], boxes_pred[:, 1])
    x2_c = np.maximum(boxes_true[:, 2], boxes_pred[:, 2])
    y2_c = np.maximum(boxes_true[:, 3], boxes_pred[:, 3])
    diag = np.sqrt((x2_c - x1_c) **2 + (y2_c - y1_c)** 2)
    
    # Calculate DIoU
    diou = iou - (dist ** 2) / (diag ** 2 + 1e-10)
    return 1 - np.mean(diou)

# 11. CIoU Loss
def ciou_loss(boxes_true, boxes_pred):
    boxes_true = np.array(boxes_true)
    boxes_pred = np.array(boxes_pred)
    
    # Calculate DIoU first
    x1_i = np.maximum(boxes_true[:, 0], boxes_pred[:, 0])
    y1_i = np.maximum(boxes_true[:, 1], boxes_pred[:, 1])
    x2_i = np.minimum(boxes_true[:, 2], boxes_pred[:, 2])
    y2_i = np.minimum(boxes_true[:, 3], boxes_pred[:, 3])
    intersection = np.maximum(0, x2_i - x1_i) * np.maximum(0, y2_i - y1_i)
    
    area_true = (boxes_true[:, 2] - boxes_true[:, 0]) * (boxes_true[:, 3] - boxes_true[:, 1])
    area_pred = (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])
    union = area_true + area_pred - intersection
    iou = intersection / (union + 1e-10)
    
    ctr_true_x = (boxes_true[:, 0] + boxes_true[:, 2]) / 2
    ctr_true_y = (boxes_true[:, 1] + boxes_true[:, 3]) / 2
    ctr_pred_x = (boxes_pred[:, 0] + boxes_pred[:, 2]) / 2
    ctr_pred_y = (boxes_pred[:, 1] + boxes_pred[:, 3]) / 2
    
    dist = np.sqrt((ctr_true_x - ctr_pred_x) **2 + (ctr_true_y - ctr_pred_y)** 2)
    
    x1_c = np.minimum(boxes_true[:, 0], boxes_pred[:, 0])
    y1_c = np.minimum(boxes_true[:, 1], boxes_pred[:, 1])
    x2_c = np.maximum(boxes_true[:, 2], boxes_pred[:, 2])
    y2_c = np.maximum(boxes_true[:, 3], boxes_pred[:, 3])
    diag = np.sqrt((x2_c - x1_c) **2 + (y2_c - y1_c)** 2)
    
    diou = iou - (dist ** 2) / (diag ** 2 + 1e-10)
    
    # Calculate aspect ratio consistency
    w_true = boxes_true[:, 2] - boxes_true[:, 0]
    h_true = boxes_true[:, 3] - boxes_true[:, 1]
    w_pred = boxes_pred[:, 2] - boxes_pred[:, 0]
    h_pred = boxes_pred[:, 3] - boxes_pred[:, 1]
    
    v = (4 / (np.pi ** 2)) * np.square(np.arctan(w_true / h_true) - np.arctan(w_pred / h_pred))
    alpha = v / (1 - iou + v + 1e-10)
    
    # Calculate CIoU
    ciou = diou - alpha * v
    return 1 - np.mean(ciou)

# 12. Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-5):
    intersection = np.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

# Visualization functions
def visualize_regression_losses():
    """Visualize L1, MSE and other regression losses"""
    plt.figure(figsize=(10, 6))
    
    # Create error values (difference between true and predicted)
    errors = np.linspace(-2, 2, 100)
    y_true = np.zeros_like(errors)  # True value is 0
    
    # Calculate different losses
    l1_losses = [l1_loss(0, e) for e in errors]
    mse_losses = [mse_loss(0, e) for e in errors]
    
    plt.plot(errors, l1_losses, label='L1 Loss', linewidth=2)
    plt.plot(errors, mse_losses, label='MSE Loss', linewidth=2)
    
    plt.xlabel('Prediction Error (y_pred - y_true)')
    plt.ylabel('Loss Value')
    plt.title('Comparison of Regression Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def visualize_classification_losses():
    """Visualize classification losses (BCE, Focal, etc.)"""
    plt.figure(figsize=(10, 6))
    
    # Create predicted probabilities for class 1
    y_pred = np.linspace(0.01, 0.99, 100)
    y_true = np.ones_like(y_pred)  # True label is 1
    
    # Calculate different losses
    bce_losses = [bce_loss(1, p) for p in y_pred]
    focal_losses = [focal_loss(1, p) for p in y_pred]
    ce_losses = [ce_loss(1, p) for p in y_pred]  # Same as BCE for binary case
    
    plt.plot(y_pred, bce_losses, label='BCE Loss', linewidth=2)
    plt.plot(y_pred, focal_losses, label='Focal Loss (γ=2)', linewidth=2)
    
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Loss Value')
    plt.title('Comparison of Classification Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def visualize_iou_based_losses():
    """Visualize IoU-based losses for bounding boxes"""
    plt.figure(figsize=(12, 8))
    
    # Create scenarios with different IoU values
    ious = np.linspace(0, 1, 100)
    
    # Calculate different IoU-based losses
    iou_losses = [1 - iou for iou in ious]  # IoU loss is 1 - IoU
    giou_losses = [1 - (iou - (0.5 - (iou * 0.3))) for iou in ious]  # Simplified GIoU
    diou_losses = [1 - (iou - (0.3 * (1 - iou))) for iou in ious]  # Simplified DIoU
    
    plt.plot(ious, iou_losses, label='IoU Loss', linewidth=2)
    plt.plot(ious, giou_losses, label='GIoU Loss', linewidth=2)
    plt.plot(ious, diou_losses, label='DIoU Loss', linewidth=2)
    
    plt.xlabel('IoU Value')
    plt.ylabel('Loss Value')
    plt.title('Comparison of IoU-based Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def visualize_segmentation_losses():
    """Visualize segmentation losses (Dice, BCE)"""
    plt.figure(figsize=(10, 6))
    
    # Create predicted probabilities
    y_pred = np.linspace(0.01, 0.99, 100)
    y_true = np.ones_like(y_pred)  # True mask is 1
    
    # Calculate different losses
    bce_losses = [bce_loss(1, p) for p in y_pred]
    dice_losses = [dice_loss(1, p) for p in y_pred]
    
    plt.plot(y_pred, bce_losses, label='BCE Loss', linewidth=2)
    plt.plot(y_pred, dice_losses, label='Dice Loss', linewidth=2)
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss Value')
    plt.title('Comparison of Segmentation Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def visualize_box_loss_scenarios():
    """Visualize how box losses handle different scenarios"""
    # Define a ground truth box [x1, y1, x2, y2]
    gt_box = [[1, 1, 3, 3]]  # Square box
    
    # Create different predicted boxes
    scenarios = {
        'Perfect Match': [[1, 1, 3, 3]],
        'Small Offset': [[1.2, 1.2, 3.2, 3.2]],
        'Large Offset': [[2, 2, 4, 4]],
        'Different Size': [[1, 1, 4, 4]],
        'No Overlap': [[4, 4, 6, 6]]
    }
    
    # Calculate different losses for each scenario
    iou_losses = []
    giou_losses = []
    diou_losses = []
    ciou_losses = []
    
    for name, pred_box in scenarios.items():
        iou_losses.append(iou_loss(gt_box, pred_box))
        giou_losses.append(giou_loss(gt_box, pred_box))
        diou_losses.append(diou_loss(gt_box, pred_box))
        ciou_losses.append(ciou_loss(gt_box, pred_box))
    
    # Create bar chart
    x = np.arange(len(scenarios))
    width = 0.2
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width/2, iou_losses, width, label='IoU Loss')
    plt.bar(x - width/2, giou_losses, width, label='GIoU Loss')
    plt.bar(x + width/2, diou_losses, width, label='DIoU Loss')
    plt.bar(x + 1.5*width/2, ciou_losses, width, label='CIoU Loss')
    
    plt.xlabel('Box Scenario')
    plt.ylabel('Loss Value')
    plt.title('Box Losses in Different Scenarios')
    plt.xticks(x, scenarios.keys(), rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run visualizations
if __name__ == "__main__":
    visualize_regression_losses()
    visualize_classification_losses()
    visualize_iou_based_losses()
    visualize_segmentation_losses()
    visualize_box_loss_scenarios()
    
    # Example usage with PyTorch implementations
    print("PyTorch Loss Equivalents:")
    print(f"L1 Loss: {nn.L1Loss.__name__}")
    print(f"MSE Loss: {nn.MSELoss.__name__}")
    print(f"BCE Loss: {nn.BCELoss.__name__}")
    print(f"BCEWithLogitsLoss: {nn.BCEWithLogitsLoss.__name__}")
    print(f"CrossEntropyLoss: {nn.CrossEntropyLoss.__name__}")
    print(f"KLDivLoss: {nn.KLDivLoss.__name__}")