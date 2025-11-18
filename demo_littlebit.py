"""
LittleBit Demo: Replicating Key Results from the Paper

This script demonstrates:
1. Layer quantization at various BPW levels (Table 13 equivalent)
2. Reconstruction error analysis
3. Visualization of activation patterns (Figure 7 equivalent)
4. Performance comparison with baseline
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from littlebit import (
    LittleBitLinear,
    calculate_rank_for_target_bpw,
    SmoothSign,
    smooth_sign
)

# Set device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")


def demo_1_basic_quantization():
    """Demonstrate basic quantization and reconstruction"""
    print("\n" + "="*80)
    print("DEMO 1: Basic Layer Quantization and Reconstruction")
    print("="*80)
    
    # Simulating a Llama2-7B MLP layer
    in_features = 11008
    out_features = 4096
    
    print(f"\nLayer dimensions: {out_features} × {in_features}")
    print(f"Original FP16 parameters: {out_features * in_features:,}")
    
    # Create a random weight matrix (simulating pre-trained weights)
    torch.manual_seed(42)
    weight = torch.randn(out_features, in_features, device=device) * 0.02
    
    # Test different BPW levels
    bpw_levels = [1.0, 0.8, 0.55, 0.3, 0.1]
    
    print(f"\n{'BPW':<8} {'Rank':<8} {'Actual BPW':<12} {'Recon. Error':<15} {'Params Ratio':<15}")
    print("-" * 75)
    
    results = []
    
    for target_bpw in bpw_levels:
        # Calculate rank for target BPW
        rank = calculate_rank_for_target_bpw(in_features, out_features, target_bpw, use_residual=True)
        
        # Initialize LittleBit layer
        lb_layer = LittleBitLinear.initialize_from_weight(weight, rank, use_residual=True)
        lb_layer = lb_layer.to(device)
        
        # Compute actual BPW
        actual_bpw = lb_layer.compute_bpw()
        
        # Test reconstruction error
        test_input = torch.randn(10, in_features, device=device)
        
        # Original output
        original_output = test_input @ weight.T
        
        # LittleBit output
        with torch.no_grad():
            lb_output = lb_layer(test_input)
        
        # Compute relative error
        recon_error = torch.norm(lb_output - original_output) / torch.norm(original_output)
        
        # Effective parameter count
        # Binary matrices: out_features*rank + in_features*rank
        # Scaling vectors: out_features + in_features + rank (per pathway)
        effective_params = 2 * (out_features * rank + in_features * rank + 
                               out_features + in_features + rank)
        params_ratio = effective_params / (out_features * in_features)
        
        print(f"{target_bpw:<8.2f} {rank:<8} {actual_bpw:<12.4f} {recon_error.item():<15.6f} {params_ratio:<15.4f}")
        
        results.append({
            'bpw': target_bpw,
            'rank': rank,
            'actual_bpw': actual_bpw,
            'error': recon_error.item(),
            'params_ratio': params_ratio
        })
    
    return results


def demo_2_activation_visualization():
    """Visualize activation patterns similar to Figure 7"""
    print("\n" + "="*80)
    print("DEMO 2: Activation Pattern Visualization")
    print("="*80)
    
    # Simulating transformer layer outputs
    seq_len = 128
    hidden_dim = 512
    
    torch.manual_seed(42)
    
    # Create a simpler layer for visualization
    in_features = 512
    out_features = 512
    weight = torch.randn(out_features, in_features, device=device) * 0.02
    
    # Create test input
    test_input = torch.randn(1, seq_len, in_features, device=device)
    
    # Reshape for linear layer
    x_flat = test_input.reshape(-1, in_features)
    
    # Full precision output
    fp_output = (x_flat @ weight.T).reshape(1, seq_len, out_features)
    
    # LittleBit with residual
    rank = calculate_rank_for_target_bpw(in_features, out_features, 0.55, use_residual=True)
    lb_with_res = LittleBitLinear.initialize_from_weight(weight, rank, use_residual=True)
    lb_with_res = lb_with_res.to(device)
    with torch.no_grad():
        lb_res_output = lb_with_res(x_flat).reshape(1, seq_len, out_features)
    
    # LittleBit without residual
    lb_no_res = LittleBitLinear.initialize_from_weight(weight, rank, use_residual=False)
    lb_no_res = lb_no_res.to(device)
    with torch.no_grad():
        lb_no_res_output = lb_no_res(x_flat).reshape(1, seq_len, out_features)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Extract activation maps
    fp_map = fp_output[0].T.cpu().numpy()
    lb_res_map = lb_res_output[0].T.cpu().numpy()
    lb_no_res_map = lb_no_res_output[0].T.cpu().numpy()
    
    # Plot
    im1 = axes[0].imshow(fp_map, aspect='auto', cmap='viridis')
    axes[0].set_title('Full Precision (FP16)')
    axes[0].set_xlabel('Token')
    axes[0].set_ylabel('Channel')
    plt.colorbar(im1, ax=axes[0], label='Value')
    
    im2 = axes[1].imshow(lb_res_map, aspect='auto', cmap='viridis')
    axes[1].set_title('LittleBit with Residual (0.55 BPW)')
    axes[1].set_xlabel('Token')
    axes[1].set_ylabel('Channel')
    plt.colorbar(im2, ax=axes[1], label='Value')
    
    im3 = axes[2].imshow(lb_no_res_map, aspect='auto', cmap='viridis')
    axes[2].set_title('LittleBit without Residual (0.55 BPW)')
    axes[2].set_xlabel('Token')
    axes[2].set_ylabel('Channel')
    plt.colorbar(im3, ax=axes[2], label='Value')
    
    plt.tight_layout()
    plt.savefig('activation_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Activation visualization saved as 'activation_visualization.png'")
    
    # Compute similarity metrics
    cos_sim_res = torch.nn.functional.cosine_similarity(
        fp_output.flatten(), lb_res_output.flatten(), dim=0
    )
    cos_sim_no_res = torch.nn.functional.cosine_similarity(
        fp_output.flatten(), lb_no_res_output.flatten(), dim=0
    )
    
    print(f"\nCosine Similarity with Full Precision:")
    print(f"  With Residual:    {cos_sim_res.item():.6f}")
    print(f"  Without Residual: {cos_sim_no_res.item():.6f}")


def demo_3_smoothsign_gradient():
    """Visualize SmoothSign activation and its gradient"""
    print("\n" + "="*80)
    print("DEMO 3: SmoothSign Activation Visualization")
    print("="*80)
    
    # Create input range
    x = torch.linspace(-0.3, 0.3, 1000, requires_grad=True)
    
    # Forward pass
    y_sign = torch.sign(x.detach())
    y_tanh = torch.tanh(100 * x.detach())
    
    # Backward pass (gradient of tanh)
    k = 100.0
    grad_tanh = k * (1 - torch.tanh(k * x.detach()) ** 2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot forward functions
    ax1.plot(x.detach().numpy(), y_sign.numpy(), 'b-', linewidth=2, label='sign(x) (Forward)')
    ax1.plot(x.detach().numpy(), y_tanh.numpy(), 'r--', linewidth=2, label='tanh(100x) (Proxy)')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Forward Pass: sign(x) and tanh(100x)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot gradient
    ax2.plot(x.detach().numpy(), grad_tanh.numpy(), 'g-', linewidth=2, 
             label='100(1 - tanh²(100x))')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('∂/∂x (assuming ∂L/∂y = 1)', fontsize=12)
    ax2.set_title('Backward Pass: Smooth Proxy Gradient', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-10, 110])
    
    plt.tight_layout()
    plt.savefig('smoothsign_activation.png', dpi=150, bbox_inches='tight')
    print("\n✓ SmoothSign visualization saved as 'smoothsign_activation.png'")


def demo_4_compression_analysis():
    """Analyze compression ratios and memory savings"""
    print("\n" + "="*80)
    print("DEMO 4: Compression Ratio and Memory Savings Analysis")
    print("="*80)
    
    # Model configurations
    models = [
        ("OPT-1.3B", 1.3e9),
        ("Llama2-7B", 7e9),
        ("Llama2-13B", 13e9),
        ("Llama2-70B", 70e9),
    ]
    
    bpw_levels = [1.0, 0.8, 0.7, 0.55, 0.3, 0.1]
    
    print(f"\nMemory Requirements (in GB):")
    print(f"\n{'Model':<15} {'FP16':<10}", end='')
    for bpw in bpw_levels:
        print(f"{bpw:.2f} BPW{'':<5}", end='')
    print()
    print("-" * 90)
    
    results_data = []
    
    for model_name, params in models:
        fp16_size = (params * 16) / (8 * 1024**3)  # Convert to GB
        row = [model_name, fp16_size]
        
        print(f"{model_name:<15} {fp16_size:<10.2f}", end='')
        
        for bpw in bpw_levels:
            compressed_size = (params * bpw) / (8 * 1024**3)
            row.append(compressed_size)
            print(f"{compressed_size:<12.2f}", end='')
        print()
        
        results_data.append(row)
    
    # Compression ratios
    print(f"\n{'Model':<15} ", end='')
    for bpw in bpw_levels:
        print(f"{bpw:.2f} BPW{'':<5}", end='')
    print("  (Compression Ratio)")
    print("-" * 90)
    
    for model_name, params in models:
        print(f"{model_name:<15} ", end='')
        for bpw in bpw_levels:
            ratio = 16.0 / bpw
            print(f"{ratio:<12.1f}×", end='')
        print()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Memory savings plot
    for i, (model_name, params) in enumerate(models):
        fp16_size = (params * 16) / (8 * 1024**3)
        sizes = [fp16_size] + [(params * bpw) / (8 * 1024**3) for bpw in bpw_levels]
        x_labels = ['FP16'] + [f'{bpw}' for bpw in bpw_levels]
        ax1.plot(x_labels, sizes, marker='o', linewidth=2, label=model_name)
    
    ax1.set_xlabel('Quantization Level (BPW)', fontsize=11)
    ax1.set_ylabel('Model Size (GB)', fontsize=11)
    ax1.set_title('Model Size vs. Quantization Level', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Compression ratio plot
    compression_ratios = [16.0 / bpw for bpw in bpw_levels]
    bpw_labels = [f'{bpw}' for bpw in bpw_levels]
    ax2.bar(bpw_labels, compression_ratios, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Bits Per Weight (BPW)', fontsize=11)
    ax2.set_ylabel('Compression Ratio (×)', fontsize=11)
    ax2.set_title('Compression Ratio vs. BPW', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bpw, ratio) in enumerate(zip(bpw_labels, compression_ratios)):
        ax2.text(i, ratio + 5, f'{ratio:.1f}×', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('compression_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Compression analysis saved as 'compression_analysis.png'")


def demo_5_kernel_efficiency():
    """Simulate kernel-level efficiency analysis"""
    print("\n" + "="*80)
    print("DEMO 5: Theoretical Computational Cost Analysis")
    print("="*80)
    
    # Llama2-7B MLP layer
    din = 11008
    dout = 4096
    
    print(f"\nAnalyzing layer: {dout} × {din}")
    print(f"Standard FP16 parameters: {dout * din:,}")
    
    # FP16 baseline cost
    fp16_macs = din * dout
    fp16_flops = 2 * fp16_macs  # Each MAC = 1 mult + 1 add
    
    print(f"\nFP16 Baseline:")
    print(f"  MACs:  {fp16_macs:,}")
    print(f"  FLOPs: {fp16_flops:,} ({fp16_flops/1e6:.1f}M)")
    
    # LittleBit costs at different BPW
    bpw_levels = [1.0, 0.8, 0.55, 0.3, 0.1]
    
    print(f"\n{'BPW':<8} {'Rank':<8} {'FLOPs (M)':<12} {'BOPs (M)':<12} {'FLOP Reduction':<15} {'Total Ops':<12}")
    print("-" * 85)
    
    for bpw in bpw_levels:
        rank = calculate_rank_for_target_bpw(din, dout, bpw, use_residual=True)
        
        # LittleBit operations (per pathway, multiply by 2 for primary + residual)
        # FLOPs: scaling operations and accumulations
        flops_per_pathway = (din + dout) * rank  # Approximately
        total_flops = 2 * flops_per_pathway
        
        # BOPs: binary matrix multiplications
        bops_per_pathway = rank * (din + dout)
        total_bops = 2 * bops_per_pathway
        
        flop_reduction = fp16_flops / total_flops
        total_ops = total_flops + total_bops  # Simplified (BOPs are cheaper)
        
        print(f"{bpw:<8.2f} {rank:<8} {total_flops/1e6:<12.1f} {total_bops/1e6:<12.1f} "
              f"{flop_reduction:<15.1f}× {total_ops/1e6:<12.1f}")
    
    print("\nNote: BOPs (Bitwise Operations) are significantly faster than FLOPs")
    print("      Actual speedup depends on hardware implementation")


def demo_6_error_analysis():
    """Analyze reconstruction error across different configurations"""
    print("\n" + "="*80)
    print("DEMO 6: Reconstruction Error Analysis")
    print("="*80)
    
    # Test with different layer sizes
    layer_configs = [
        (4096, 4096, "Square Layer (4096×4096)"),
        (4096, 11008, "MLP Up-projection (4096×11008)"),
        (11008, 4096, "MLP Down-projection (11008×4096)"),
    ]
    
    bpw_levels = [1.0, 0.8, 0.55, 0.3, 0.1]
    
    torch.manual_seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (out_f, in_f, name) in enumerate(layer_configs):
        print(f"\n{name}:")
        print(f"{'BPW':<8} {'With Residual':<15} {'Without Residual':<15} {'Improvement':<12}")
        print("-" * 55)
        
        errors_with = []
        errors_without = []
        
        # Create weight matrix
        weight = torch.randn(out_f, in_f, device=device) * 0.02
        test_input = torch.randn(10, in_f, device=device)
        original_output = test_input @ weight.T
        
        for bpw in bpw_levels:
            rank = calculate_rank_for_target_bpw(in_f, out_f, bpw, use_residual=True)
            
            # With residual
            lb_with = LittleBitLinear.initialize_from_weight(weight, rank, use_residual=True)
            lb_with = lb_with.to(device)
            with torch.no_grad():
                out_with = lb_with(test_input)
            error_with = (torch.norm(out_with - original_output) / torch.norm(original_output)).item()
            errors_with.append(error_with)
            
            # Without residual
            lb_without = LittleBitLinear.initialize_from_weight(weight, rank, use_residual=False)
            lb_without = lb_without.to(device)
            with torch.no_grad():
                out_without = lb_without(test_input)
            error_without = (torch.norm(out_without - original_output) / torch.norm(original_output)).item()
            errors_without.append(error_without)
            
            improvement = (error_without - error_with) / error_without * 100
            print(f"{bpw:<8.2f} {error_with:<15.6f} {error_without:<15.6f} {improvement:>10.1f}%")
        
        # Plot
        axes[idx].plot(bpw_levels, errors_with, marker='o', linewidth=2, label='With Residual')
        axes[idx].plot(bpw_levels, errors_without, marker='s', linewidth=2, label='Without Residual')
        axes[idx].set_xlabel('BPW', fontsize=11)
        axes[idx].set_ylabel('Relative Error', fontsize=11)
        axes[idx].set_title(name, fontsize=11)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale('log')
        axes[idx].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Error analysis saved as 'error_analysis.png'")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("LittleBit: Extreme Low-Bit Quantization for LLMs")
    print("Replicating Key Results from arXiv:2506.13771v2")
    print("="*80)
    
    # Run demonstrations
    demo_1_basic_quantization()
    demo_2_activation_visualization()
    demo_3_smoothsign_gradient()
    demo_4_compression_analysis()
    demo_5_kernel_efficiency()
    demo_6_error_analysis()
    
    print("\n" + "="*80)
    print("All demonstrations completed successfully!")
    print("Generated visualizations:")
    print("  1. activation_visualization.png")
    print("  2. smoothsign_activation.png")
    print("  3. compression_analysis.png")
    print("  4. error_analysis.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
