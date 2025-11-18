"""
LittleBit: Extreme Low-Bit Quantization for Large Language Models
Implementation based on the paper arXiv:2506.13771v2

This module implements the core components of LittleBit:
1. Dual-SVID initialization
2. Binary factorization with multi-scale compensation
3. SmoothSign activation for training
4. Residual compensation mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SmoothSign(torch.autograd.Function):
    """
    SmoothSign activation function with smooth gradient approximation.
    Forward: sign(x)
    Backward: gradient of tanh(k*x) where k=100
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        k = 100.0
        # Gradient of tanh(kx) = k * (1 - tanh^2(kx))
        grad_input = grad_output * k * (1 - torch.tanh(k * x) ** 2)
        return grad_input


def smooth_sign(x):
    """Apply SmoothSign activation"""
    return SmoothSign.apply(x)


class LittleBitLinear(nn.Module):
    """
    LittleBit Linear Layer implementing Dual-SVID with multi-scale compensation.
    
    Replaces a standard linear layer W with:
    - Primary pathway: diag(h) @ U_sign @ diag(l) @ V_sign^T @ diag(g)
    - Residual pathway: diag(h_res) @ U_sign_res @ diag(l_res) @ V_sign_res^T @ diag(g_res)
    
    Where U_sign, V_sign are binary matrices {-1, +1}
    and h, l, g are FP16 scaling vectors
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        use_residual: bool = True,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.use_residual = use_residual
        
        # Primary pathway parameters
        # Binary matrices (stored as float for gradients, but quantized in forward)
        self.U_primary = nn.Parameter(torch.randn(out_features, rank))
        self.V_primary = nn.Parameter(torch.randn(in_features, rank))
        
        # Scaling vectors (FP16)
        self.h_primary = nn.Parameter(torch.ones(out_features))
        self.l_primary = nn.Parameter(torch.ones(rank))
        self.g_primary = nn.Parameter(torch.ones(in_features))
        
        # Residual pathway (if enabled)
        if use_residual:
            self.U_residual = nn.Parameter(torch.randn(out_features, rank))
            self.V_residual = nn.Parameter(torch.randn(in_features, rank))
            self.h_residual = nn.Parameter(torch.ones(out_features))
            self.l_residual = nn.Parameter(torch.ones(rank))
            self.g_residual = nn.Parameter(torch.ones(in_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    @staticmethod
    def initialize_from_weight(
        weight: torch.Tensor,
        rank: int,
        use_residual: bool = True
    ) -> 'LittleBitLinear':
        """
        Initialize LittleBit layer from a full-precision weight matrix using Dual-SVID.
        
        Args:
            weight: Original weight matrix of shape (out_features, in_features)
            rank: Latent dimension for factorization
            use_residual: Whether to use residual compensation
        
        Returns:
            Initialized LittleBitLinear layer
        """
        out_features, in_features = weight.shape
        layer = LittleBitLinear(in_features, out_features, rank, use_residual, bias=False)
        
        # Perform SVD on original weight
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        
        # Take top-r components for primary pathway
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = Vt[:rank, :].T
        
        # Initialize primary pathway
        # Decompose into signs and magnitudes
        U_sign = torch.sign(U_r)
        U_mag = torch.abs(U_r)
        V_sign = torch.sign(V_r)
        V_mag = torch.abs(V_r)
        
        # Initialize scaling vectors using magnitude means
        layer.h_primary.data = U_mag.mean(dim=1)
        layer.g_primary.data = V_mag.mean(dim=1)
        layer.l_primary.data = S_r
        
        # Initialize binary matrices
        layer.U_primary.data = U_sign
        layer.V_primary.data = V_sign
        
        # Initialize residual pathway if enabled
        if use_residual:
            # Compute primary reconstruction
            W_primary = layer._compute_weight_primary()
            residual = weight - W_primary
            
            # SVD on residual
            U_res, S_res, Vt_res = torch.linalg.svd(residual, full_matrices=False)
            U_res_r = U_res[:, :rank]
            S_res_r = S_res[:rank]
            V_res_r = Vt_res[:rank, :].T
            
            # Decompose residual into signs and magnitudes
            U_res_sign = torch.sign(U_res_r)
            U_res_mag = torch.abs(U_res_r)
            V_res_sign = torch.sign(V_res_r)
            V_res_mag = torch.abs(V_res_r)
            
            layer.h_residual.data = U_res_mag.mean(dim=1)
            layer.g_residual.data = V_res_mag.mean(dim=1)
            layer.l_residual.data = S_res_r
            layer.U_residual.data = U_res_sign
            layer.V_residual.data = V_res_sign
        
        return layer
    
    def _compute_weight_primary(self) -> torch.Tensor:
        """Compute the primary weight matrix"""
        U_sign = smooth_sign(self.U_primary)
        V_sign = smooth_sign(self.V_primary)
        
        # W_primary = diag(h) @ U_sign @ diag(l) @ V_sign^T @ diag(g)
        W = torch.diag(self.h_primary) @ U_sign @ torch.diag(self.l_primary) @ V_sign.T @ torch.diag(self.g_primary)
        return W
    
    def _compute_weight_residual(self) -> torch.Tensor:
        """Compute the residual weight matrix"""
        if not self.use_residual:
            return 0
        
        U_sign = smooth_sign(self.U_residual)
        V_sign = smooth_sign(self.V_residual)
        
        # W_residual = diag(h_res) @ U_sign_res @ diag(l_res) @ V_sign_res^T @ diag(g_res)
        W = torch.diag(self.h_residual) @ U_sign @ torch.diag(self.l_residual) @ V_sign.T @ torch.diag(self.g_residual)
        return W
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using efficient factorized computation.
        
        Following Eq. (12) from the paper:
        Y = ((((X ⊙ g) @ V_sign) ⊙ l) @ U_sign^T) ⊙ h
        """
        # Primary pathway
        # Step 1: X ⊙ g (element-wise multiplication with broadcasting)
        out = x * self.g_primary.unsqueeze(0)
        
        # Step 2: @ V_sign (matmul with binary matrix)
        V_sign = smooth_sign(self.V_primary)
        out = out @ V_sign
        
        # Step 3: ⊙ l
        out = out * self.l_primary.unsqueeze(0)
        
        # Step 4: @ U_sign^T
        U_sign = smooth_sign(self.U_primary)
        out = out @ U_sign.T
        
        # Step 5: ⊙ h
        out = out * self.h_primary.unsqueeze(0)
        
        # Add residual pathway if enabled
        if self.use_residual:
            out_res = x * self.g_residual.unsqueeze(0)
            V_sign_res = smooth_sign(self.V_residual)
            out_res = out_res @ V_sign_res
            out_res = out_res * self.l_residual.unsqueeze(0)
            U_sign_res = smooth_sign(self.U_residual)
            out_res = out_res @ U_sign_res.T
            out_res = out_res * self.h_residual.unsqueeze(0)
            out = out + out_res
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def compute_bpw(self) -> float:
        """
        Compute average bits per weight (BPW) for this layer.
        
        Following Eq. (16) from the paper:
        BPW = [2r(d_out + d_in) + 32(d_out + d_in) + 32r] / (d_out * d_in)
        
        Where:
        - Binary matrices: 1 bit per element
        - Scaling vectors: 16 bits per element
        - Factor of 2 accounts for both primary and residual pathways
        """
        d_out = self.out_features
        d_in = self.in_features
        r = self.rank
        
        if self.use_residual:
            factor = 2
        else:
            factor = 1
        
        total_bits = factor * (2 * r * (d_out + d_in) + 32 * (d_out + d_in) + 32 * r)
        total_params = d_out * d_in
        
        return total_bits / total_params


def calculate_rank_for_target_bpw(
    in_features: int,
    out_features: int,
    target_bpw: float,
    use_residual: bool = True
) -> int:
    """
    Calculate the latent rank r needed to achieve a target BPW.
    
    Following Eq. (17) from the paper:
    r = [(b × d_out × d_in) - 32(d_out + d_in)] / [2(d_out + d_in) + 32]
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        target_bpw: Target bits per weight
        use_residual: Whether residual compensation is used (affects calculation)
    
    Returns:
        Calculated rank (integer)
    """
    d_out = out_features
    d_in = in_features
    b = target_bpw
    
    numerator = (b * d_out * d_in) - 32 * (d_out + d_in)
    denominator = 2 * (d_out + d_in) + 32
    
    r = numerator / denominator
    
    # Round to nearest integer
    r = int(round(r))
    
    # Ensure r is positive and reasonable
    r = max(1, min(r, min(d_out, d_in)))
    
    return r


class LittleBitConfig:
    """Configuration for LittleBit quantization"""
    
    def __init__(
        self,
        target_bpw: float = 0.55,
        use_residual: bool = True,
        kv_rank_multiplier: int = 1,
        learning_rate: float = 8e-5,
        num_epochs: int = 5,
        batch_size: int = 8,
        temperature: float = 1.0
    ):
        self.target_bpw = target_bpw
        self.use_residual = use_residual
        self.kv_rank_multiplier = kv_rank_multiplier  # For GQA/MQA models
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.temperature = temperature  # For knowledge distillation


def compute_actual_bpw(ranks: dict, shapes: dict, use_residual: bool = True) -> float:
    """
    Compute the average BPW across all layers in a model.
    
    Args:
        ranks: Dictionary mapping layer names to their ranks
        shapes: Dictionary mapping layer names to (out_features, in_features)
        use_residual: Whether residual compensation is used
    
    Returns:
        Average bits per weight across all layers
    """
    total_bits = 0
    total_params = 0
    
    factor = 2 if use_residual else 1
    
    for layer_name, (d_out, d_in) in shapes.items():
        r = ranks.get(layer_name, 0)
        
        # Calculate bits for this layer
        layer_bits = factor * (2 * r * (d_out + d_in) + 32 * (d_out + d_in) + 32 * r)
        layer_params = d_out * d_in
        
        total_bits += layer_bits
        total_params += layer_params
    
    return total_bits / total_params if total_params > 0 else 0


if __name__ == "__main__":
    # Example usage and verification
    print("LittleBit Implementation Test\n")
    print("=" * 60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Example 1: Single layer quantization
    print("\n1. Single Layer Quantization Example")
    print("-" * 60)
    
    # Create a random weight matrix
    in_features = 4096
    out_features = 4096
    target_bpw = 0.55
    
    print(f"Original layer: {out_features} × {in_features}")
    print(f"Target BPW: {target_bpw}")
    
    # Calculate required rank
    rank = calculate_rank_for_target_bpw(in_features, out_features, target_bpw, use_residual=True)
    print(f"Calculated rank: {rank}")
    
    # Create random weight
    weight = torch.randn(out_features, in_features, device=device) * 0.02
    
    # Initialize LittleBit layer
    lb_layer = LittleBitLinear.initialize_from_weight(weight, rank, use_residual=True)
    lb_layer = lb_layer.to(device)
    
    # Compute actual BPW
    actual_bpw = lb_layer.compute_bpw()
    print(f"Actual BPW: {actual_bpw:.4f}")
    
    # Test forward pass
    x = torch.randn(1, in_features, device=device)
    output = lb_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compare with original
    original_output = x @ weight.T
    error = torch.norm(output - original_output) / torch.norm(original_output)
    print(f"Relative reconstruction error: {error.item():.4f}")
    
    # Example 2: Calculate ranks for different BPW targets
    print("\n2. Rank Calculation for Various BPW Targets")
    print("-" * 60)
    
    layer_configs = [
        (4096, 4096, "Llama2-7B ATTN-like"),
        (4096, 11008, "Llama2-7B MLP-like"),
        (8192, 28672, "Llama2-70B MLP-like")
    ]
    
    bpw_targets = [1.0, 0.8, 0.7, 0.55, 0.3, 0.1]
    
    for out_f, in_f, name in layer_configs:
        print(f"\n{name} ({out_f} × {in_f}):")
        print(f"{'BPW':<8} {'Rank':<8} {'Actual BPW':<12}")
        print("-" * 30)
        
        for bpw in bpw_targets:
            r = calculate_rank_for_target_bpw(in_f, out_f, bpw, use_residual=True)
            
            # Create a dummy layer to compute actual BPW
            dummy_layer = LittleBitLinear(in_f, out_f, r, use_residual=True, bias=False)
            actual = dummy_layer.compute_bpw()
            
            print(f"{bpw:<8.2f} {r:<8} {actual:<12.4f}")
    
    # Example 3: Memory savings
    print("\n3. Memory Savings Analysis")
    print("-" * 60)
    
    # Llama2-7B approximate layer sizes
    total_params = 7e9  # 7 billion parameters
    fp16_size_gb = (total_params * 16) / (8 * 1024**3)  # Convert bits to GB
    
    print(f"Original FP16 model size: {fp16_size_gb:.2f} GB")
    print(f"\n{'BPW':<8} {'Size (GB)':<12} {'Compression':<12} {'Memory Saved'}")
    print("-" * 50)
    
    for bpw in bpw_targets:
        compressed_size_gb = (total_params * bpw) / (8 * 1024**3)
        compression_ratio = 16.0 / bpw
        memory_saved = fp16_size_gb - compressed_size_gb
        
        print(f"{bpw:<8.2f} {compressed_size_gb:<12.2f} {compression_ratio:<12.1f}× {memory_saved:>6.2f} GB")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
