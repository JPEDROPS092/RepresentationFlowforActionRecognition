import torch
import torch.nn as nn
import torch.nn.functional as F


class RepFlowLayer(nn.Module):
    """
    Representation Flow Layer for capturing temporal dynamics in video sequences.
    Optimized for gesture recognition tasks.
    """
    
    def __init__(self, input_channels, flow_channels=None, kernel_size=3, stride=1, padding=1):
        super(RepFlowLayer, self).__init__()
        
        if flow_channels is None:
            flow_channels = input_channels // 2
            
        self.input_channels = input_channels
        self.flow_channels = flow_channels
        
        # Flow estimation network
        self.flow_conv1 = nn.Conv3d(input_channels * 2, 64, kernel_size=(1, 3, 3), 
                                   stride=(1, 1, 1), padding=(0, 1, 1))
        self.flow_conv2 = nn.Conv3d(64, 32, kernel_size=(1, 3, 3), 
                                   stride=(1, 1, 1), padding=(0, 1, 1))
        self.flow_conv3 = nn.Conv3d(32, flow_channels, kernel_size=(1, 3, 3), 
                                   stride=(1, 1, 1), padding=(0, 1, 1))
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(32)
        
    def forward(self, x):
        """
        Forward pass of the Representation Flow Layer.
        
        Args:
            x: Input tensor of shape (N, C, T, H, W)
            
        Returns:
            flow: Flow features of shape (N, flow_channels, T-1, H, W)
        """
        N, C, T, H, W = x.shape
        
        if T < 2:
            raise ValueError("Temporal dimension must be at least 2 for flow computation")
        
        # Compute frame pairs for flow estimation
        flows = []
        
        for t in range(T - 1):
            # Get consecutive frames
            frame1 = x[:, :, t:t+1, :, :]     # (N, C, 1, H, W)
            frame2 = x[:, :, t+1:t+2, :, :]   # (N, C, 1, H, W)
            
            # Concatenate frames along channel dimension
            frame_pair = torch.cat([frame1, frame2], dim=1)  # (N, 2C, 1, H, W)
            
            # Compute flow for this pair
            flow = self.relu(self.bn1(self.flow_conv1(frame_pair)))
            flow = self.relu(self.bn2(self.flow_conv2(flow)))
            flow = self.tanh(self.flow_conv3(flow))  # (N, flow_channels, 1, H, W)
            
            flows.append(flow)
        
        # Concatenate all flows along temporal dimension
        if len(flows) == 1:
            rep_flow = flows[0]
        else:
            rep_flow = torch.cat(flows, dim=2)  # (N, flow_channels, T-1, H, W)
        
        return rep_flow


class AdaptiveRepFlowLayer(nn.Module):
    """
    Adaptive Representation Flow Layer with attention mechanism for gesture recognition.
    """
    
    def __init__(self, input_channels, flow_channels=None, temporal_attention=True):
        super(AdaptiveRepFlowLayer, self).__init__()
        
        if flow_channels is None:
            flow_channels = input_channels // 2
            
        self.input_channels = input_channels
        self.flow_channels = flow_channels
        self.temporal_attention = temporal_attention
        
        # Base flow layer
        self.rep_flow = RepFlowLayer(input_channels, flow_channels)
        
        # Temporal attention mechanism
        if temporal_attention:
            self.temporal_attn = nn.Sequential(
                nn.Conv3d(flow_channels, flow_channels // 4, kernel_size=(3, 1, 1), 
                         stride=(1, 1, 1), padding=(1, 0, 0)),
                nn.ReLU(inplace=True),
                nn.Conv3d(flow_channels // 4, 1, kernel_size=(3, 1, 1), 
                         stride=(1, 1, 1), padding=(1, 0, 0)),
                nn.Sigmoid()
            )
        
        # Spatial attention mechanism
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(flow_channels, 1, kernel_size=(1, 7, 7), 
                     stride=(1, 1, 1), padding=(0, 3, 3)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass with attention mechanisms.
        
        Args:
            x: Input tensor of shape (N, C, T, H, W)
            
        Returns:
            attended_flow: Attention-weighted flow features
        """
        # Compute base representation flow
        flow = self.rep_flow(x)  # (N, flow_channels, T-1, H, W)
        
        # Apply temporal attention
        if self.temporal_attention and flow.shape[2] > 1:
            temp_attn = self.temporal_attn(flow)  # (N, 1, T-1, H, W)
            flow = flow * temp_attn
        
        # Apply spatial attention
        spatial_attn = self.spatial_attn(flow)  # (N, 1, T-1, H, W)
        attended_flow = flow * spatial_attn
        
        return attended_flow