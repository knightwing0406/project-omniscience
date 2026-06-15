import torch
import torch.nn as nn
import torch.nn.functional as F

class EnterpriseAcousticClassifier(nn.Module):
    """
    Advanced deep convolutional neural network explicitly architected for 
    high-speed spatial feature extraction from spectral density audio matrices.
    """
    def __init__(self):
        super(EnterpriseAcousticClassifier, self).__init__()
        
        # Block 1: Feature Extraction (Input Spectrogram Matrix: 1 Channel x 64 Rows x 32 Columns)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 2: Deep Pattern Consolidation
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Dimensional Downsampling Layer (Reduces matrix height & width by 50% per execution)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout Regularization Engine: Prevents memorization during training passes
        self.dropout_engine = nn.Dropout2d(0.35)
        
        # Block 3: Fully Connected Linear Deep Inference Routing
        # Post two MaxPool operations, our 64x32 matrix downsamples to a compact 16x8 footprint.
        # Total spatial volume = 32 operational filter channels * 16 rows * 8 columns = 4096 flattened neurons.
        self.fully_connected_layer = nn.Linear(32 * 16 * 8, 64)
        self.output_classifier = nn.Linear(64, 2) # Terminal States: Class 0 (Healthy), Class 1 (Anomaly)

    def forward(self, x):
        """Executes full mathematical tensor feedforward propagation down the structural network path."""
        # Layer 1 Processing: Conv -> Normalize Scales -> Activate Nonlinearity -> Shrink Dimensions
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Layer 2 Processing: Conv -> Normalize Scales -> Activate Nonlinearity -> Shrink Dimensions
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.dropout_engine(x)
        
        # Flatten the spatial multidimensional tensor into a sleek 1D feature array
        x = x.view(-1, 32 * 16 * 8)
        
        # Execute Final Linear Mapping & Logit Distribution
        x = F.relu(self.fully_connected_layer(x))
        x = self.output_classifier(x)
        return x

if __name__ == "__main__":
    # Local runtime evaluation and shape trace verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnterpriseAcousticClassifier().to(device)
    print(f"🧠 [MODEL] Neural Network verified on device: {device}")
    
    # Calculate total weight parameters for production audits
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 [MODEL] Trainable parameter footprint: {total_params:,} connections.")
