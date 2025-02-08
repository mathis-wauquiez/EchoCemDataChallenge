import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Compute Multi-Class Dice Loss.
        
        Args:
            predictions (torch.Tensor): Predicted segmentation logits (shape: [B, C, H, W]).
            targets (torch.Tensor): Ground truth segmentation masks (shape: [B, H, W]).
        
        Returns:
            torch.Tensor: Multi-Class Dice Loss value.
        """
        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)

        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()

        # Compute Dice Loss for each class
        dice_loss = 0.0
        for class_idx in range(num_classes):
            pred = predictions[:, class_idx, :, :].contiguous().view(-1)
            target = targets[:, class_idx, :, :].contiguous().view(-1)

            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1.0 - dice

        # Average Dice Loss across classes
        return dice_loss / num_classes

# Example usage
if __name__ == "__main__":
    # Create dummy predictions and targets
    predictions = torch.randn(4, 3, 256, 256)  # Batch of 4, 3 classes, 256x256 images
    targets = torch.randint(0, 3, (4, 256, 256))  # Ground truth masks with class indices

    # Initialize Multi-Class Dice Loss
    dice_loss = MultiClassDiceLoss()

    # Compute loss
    loss = dice_loss(predictions, targets)
    print(f"Multi-Class Dice Loss: {loss.item()}")