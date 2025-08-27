import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video.r2plus1d import R2Plus1D_18_Weights
from rep_flow_layer import RepFlowLayer, AdaptiveRepFlowLayer


class GestureRecognitionModel(nn.Module):
    """
    Gesture Recognition Model with Representation Flow integration.
    Uses a pre-trained foundation model with inserted flow layers.
    """
    
    def __init__(self, 
                 num_classes=20,  # Common gesture recognition datasets have 10-25 classes
                 flow_channels=64,
                 use_adaptive_flow=True,
                 freeze_backbone=True,
                 dropout_rate=0.5):
        super(GestureRecognitionModel, self).__init__()
        
        # Load pre-trained R(2+1)D model as foundation
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.backbone = models.video.r2plus1d_18(weights=weights)
        
        # Extract components of the backbone
        self.stem = self.backbone.stem
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.avgpool = self.backbone.avgpool
        
        # Representation Flow layers at different levels
        if use_adaptive_flow:
            self.flow_layer1 = AdaptiveRepFlowLayer(64, flow_channels//2)   # After layer1: 64 channels
            self.flow_layer2 = AdaptiveRepFlowLayer(128, flow_channels)     # After layer2: 128 channels
        else:
            self.flow_layer1 = RepFlowLayer(64, flow_channels//2)
            self.flow_layer2 = RepFlowLayer(128, flow_channels)
        
        # Multi-scale flow fusion
        self.flow_fusion = nn.Sequential(
            nn.Conv3d(flow_channels//2 + flow_channels, flow_channels, 
                     kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(flow_channels),
            nn.ReLU(inplace=True)
        )
        
        # Gesture-specific attention mechanism
        self.gesture_attention = nn.MultiheadAttention(
            embed_dim=512 + flow_channels,  # backbone features + flow features
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + flow_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze the pre-trained backbone parameters."""
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass for gesture recognition.
        
        Args:
            x: Input tensor of shape (N, C, T, H, W)
            
        Returns:
            logits: Classification logits of shape (N, num_classes)
        """
        N, C, T, H, W = x.shape
        
        # Backbone feature extraction
        x = self.stem(x)
        
        # Layer 1 with flow computation
        x1 = self.layer1(x)
        if x1.shape[2] >= 2:  # Ensure temporal dimension is sufficient
            flow1 = self.flow_layer1(x1)  # (N, flow_channels//2, T-k, H', W')
        else:
            flow1 = None
        
        # Layer 2 with flow computation
        x2 = self.layer2(x1)
        if x2.shape[2] >= 2:
            flow2 = self.flow_layer2(x2)  # (N, flow_channels, T-k, H'', W'')
        else:
            flow2 = None
        
        # Continue backbone processing
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        backbone_features = self.avgpool(x4)  # (N, 512, 1, 1, 1)
        backbone_features = backbone_features.flatten(1)  # (N, 512)
        
        # Process and fuse flow features
        if flow1 is not None and flow2 is not None:
            # Resize flows to same spatial dimensions
            _, _, T_flow, H_flow, W_flow = flow2.shape
            if flow1.shape[2:] != flow2.shape[2:]:
                flow1_resized = F.interpolate(flow1, size=(T_flow, H_flow, W_flow), 
                                            mode='trilinear', align_corners=False)
            else:
                flow1_resized = flow1
            
            # Concatenate and fuse flows
            combined_flow = torch.cat([flow1_resized, flow2], dim=1)
            fused_flow = self.flow_fusion(combined_flow)
            
            # Global average pooling for flow features
            flow_features = F.adaptive_avg_pool3d(fused_flow, (1, 1, 1))
            flow_features = flow_features.flatten(1)  # (N, flow_channels)
        
        elif flow2 is not None:
            flow_features = F.adaptive_avg_pool3d(flow2, (1, 1, 1))
            flow_features = flow_features.flatten(1)
        else:
            # Fallback if no flow can be computed
            flow_features = torch.zeros(N, self.flow_layer2.flow_channels).to(x.device)
        
        # Combine backbone and flow features
        combined_features = torch.cat([backbone_features, flow_features], dim=1)
        
        # Apply gesture-specific attention (treating as sequence of length 1)
        combined_features = combined_features.unsqueeze(1)  # (N, 1, feature_dim)
        attended_features, _ = self.gesture_attention(
            combined_features, combined_features, combined_features
        )
        attended_features = attended_features.squeeze(1)  # (N, feature_dim)
        
        # Apply dropout and classify
        attended_features = self.dropout(attended_features)
        logits = self.classifier(attended_features)
        
        return logits


class LightweightGestureModel(nn.Module):
    """
    Lightweight version for real-time gesture recognition.
    """
    
    def __init__(self, num_classes=20, flow_channels=32):
        super(LightweightGestureModel, self).__init__()
        
        # Lightweight backbone (MobileNet3D-inspired)
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv3d(3, 16, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Depthwise separable blocks
            self._depthwise_block(16, 32, stride=(1, 2, 2)),
            self._depthwise_block(32, 64, stride=(1, 2, 2)),
            self._depthwise_block(64, 128, stride=(2, 2, 2)),
        )
        
        # Single flow layer
        self.flow_layer = RepFlowLayer(64, flow_channels)  # Insert after 2nd block
        
        # Final pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(128 + flow_channels, num_classes)
    
    def _depthwise_block(self, in_channels, out_channels, stride=(1, 1, 1)):
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), 
                     stride=stride, padding=(1, 1, 1), groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = []
        
        # Process through backbone and collect intermediate features
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 4:  # After 2nd depthwise block (64 channels)
                if x.shape[2] >= 2:
                    flow = self.flow_layer(x)
                    flow_pooled = self.global_pool(flow).flatten(1)
                else:
                    flow_pooled = torch.zeros(x.size(0), self.flow_layer.flow_channels).to(x.device)
        
        # Final backbone features
        backbone_pooled = self.global_pool(x).flatten(1)
        
        # Combine features
        combined = torch.cat([backbone_pooled, flow_pooled], dim=1)
        logits = self.classifier(combined)
        
        return logits


def create_gesture_model(model_type='full', num_classes=20, **kwargs):
    """
    Factory function to create gesture recognition models.
    
    Args:
        model_type: 'full', 'lightweight'
        num_classes: Number of gesture classes
        **kwargs: Additional model parameters
    
    Returns:
        model: Gesture recognition model
    """
    if model_type == 'full':
        return GestureRecognitionModel(num_classes=num_classes, **kwargs)
    elif model_type == 'lightweight':
        return LightweightGestureModel(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_gesture_model('full', num_classes=20)
    model.to(device)
    
    # Test forward pass
    test_input = torch.randn(2, 3, 16, 112, 112).to(device)  # (N, C, T, H, W)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")