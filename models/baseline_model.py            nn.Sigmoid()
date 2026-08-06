import torch
import torch.nn as nn
from torchvision import models

class SingleBranchBaseline(nn.Module):
    def __init__(self):
        super(SingleBranchBaseline, self).__init__()
        
        # Baseline, Branch A (EfficientNet)
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Binary classification (Real vs Fake)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, full_face):
        
        return self.backbone(full_face)

baseline_model = SingleBranchBaseline()
print("Baseline Model (Branch A in isolation) Created Successfully!")
