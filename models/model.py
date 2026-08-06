import torch
import torch.nn as nn
from torchvision import models

class DualBranchDeepfakeDetector(nn.Module):
    def __init__(self):
        super(DualBranchDeepfakeDetector, self).__init__()
        
        # 1. Branch A (CNN) - Global Features
        self.branch_a_cnn = models.efficientnet_b0(pretrained=True)
        self.branch_a_cnn.classifier = nn.Identity() 
        
        # 2. Branch B (Eye Landmark) - Eye Biological Features
        self.branch_b_eye = models.resnet18(pretrained=True)
        self.branch_b_eye.fc = nn.Identity()
        
        # 3. Fusion Layer (Branch A + Branch B)
        combined_features_dim = 1280 + 512  
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 4. Classifier (Real/Fake Label & Explanation Output)
        self.label_classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid() # Real / Fake Label (0 - 1)
        )
        
        self.explanation_head = nn.Linear(256, 64) # Explanation Output

    def forward(self, full_face, eye_region):
        
        feat_a = self.branch_a_cnn(full_face)
        
        
        feat_b = self.branch_b_eye(eye_region)
        
        # Fusion Layer 
        combined = torch.cat((feat_a, feat_b), dim=1)
        fused = self.fusion_layer(combined)
        
        # Classifier 
        label_output = self.label_classifier(fused)
        explanation_output = self.explanation_head(fused)
        
        return label_output, explanation_output


dual_model = DualBranchDeepfakeDetector()
print("Dual-Branch Architecture Code is Ready and Matches the Diagram!")
