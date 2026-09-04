import numpy as np
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Define the index range [-20, 20]
n = np.arange(-20, 21)

# ==================== SECTION 1: Generate Signals ====================

# 1. Delta function
delta = np.zeros_like(n, dtype=float)
delta[n == 0] = 1.0

# 2. Exponential signal with negative "a" (a = -0.1)
a = -0.1
exponential = np.exp(a * n)

# 3. Ramp signal
ramp = np.maximum(0, n)

# 4. Sinusoidal wave with 0.05 Hz and unit amplitude
freq = 0.05
amplitude = 1.0
sinusoidal = amplitude * np.sin(2 * np.pi * freq * n)

# Save individual signal plots
signals = {
    'delta': delta,
    'exponential': exponential,
    'ramp': ramp,
    'sinusoidal': sinusoidal
}

for name, signal in signals.items():
    plt.figure(figsize=(10, 4))
    plt.stem(n, signal)
    plt.title(f'{name.capitalize()} Signal')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(f'outputs/1_{name}_signal.png', dpi=150, bbox_inches='tight')
    plt.close()

# ==================== SECTION 2: Add Noise and Plot ====================

np.random.seed(42)  # For reproducibility

noisy_signals = {}
for name, signal in signals.items():
    noise = np.random.uniform(0, 1, size=len(n))
    noisy_signals[name] = signal + noise

# Plot all noisy signals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Noisy Signals', fontsize=16)

for ax, (name, signal) in zip(axes.flatten(), noisy_signals.items()):
    ax.stem(n, signal)
    ax.set_title(f'{name.capitalize()} Signal + Noise')
    ax.set_xlabel('n')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

plt.tight_layout()
plt.savefig('outputs/2_all_noisy_signals.png', dpi=150, bbox_inches='tight')
plt.close()

# Save individual noisy signal plots
for name, signal in noisy_signals.items():
    plt.figure(figsize=(10, 4))
    plt.stem(n, signal)
    plt.title(f'{name.capitalize()} Signal + Noise')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(f'outputs/2_{name}_noisy.png', dpi=150, bbox_inches='tight')
    plt.close()

# ==================== SECTION 3: Convolution Functions ====================

def convolve(x, h):
    """
    Perform convolution using summation formula (not using np.convolve).
    
    Formula: y[n] = sum_{k} x[k] * h[n-k]
    
    Parameters:
    - x: input signal
    - h: impulse response (kernel)
    
    Returns:
    - y: convolved signal
    """
    nx = len(x)
    nh = len(h)
    ny = nx + nh - 1
    
    # Pad x with zeros
    x_padded = np.pad(x, (nh-1, nh-1), mode='constant')
    
    # Flip h for convolution
    h_flipped = h[::-1]
    
    # Perform convolution using sliding window
    y = np.zeros(ny)
    for i in range(ny):
        y[i] = np.sum(x_padded[i:i+nh] * h_flipped)
    
    return y

# Define the impulse responses (kernels)
kernels = {
    'A': np.array([0, 0, 0, 1, 0, 0, 0]),      # Identity-like (center tap)
    'B': np.array([1, 1, 1, 1, 1, 1, 1]),       # Moving average
    'C': np.array([-1, -1, -1, -1, -1, -1, -1]), # Inverted moving average
    'D': np.array([0, 1, 0, -1, 0, 1, 0]),      # Differentiator-like
    'E': np.array([1, -1, 1, -1, 1, -1, 1]),    # High-pass filter
    'F': np.array([0, 0.2, 0.4, 0.5, 0.4, 0.2, 0]) # Gaussian-like smoothing
}

# ==================== SECTION 3 & 4: Convolution with All Kernels ====================

# Use the noisy signals for convolution
for kernel_name, kernel in kernels.items():
    # Create output directory for this kernel
    kernel_dir = f'outputs/kernel_{kernel_name}'
    os.makedirs(kernel_dir, exist_ok=True)
    
    # Convolve each noisy signal with this kernel
    for name, signal in noisy_signals.items():
        convolved = convolve(signal, kernel)
        
        # Create time indices for convolved signal
        n_conv = np.arange(len(convolved))
        
        # Plot original noisy signal and convolved result
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Convolution with Kernel {kernel_name}: {name.capitalize()} Signal', fontsize=14)
        
        axes[0].stem(n, signal)
        axes[0].set_title(f'Original Noisy {name.capitalize()} Signal')
        axes[0].set_xlabel('n')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        axes[1].stem(n_conv, convolved)
        axes[1].set_title(f'Convolved Signal with Kernel {kernel_name}')
        axes[1].set_xlabel('n')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{kernel_dir}/convolution_{name}_kernel{kernel_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create comparison plot for all signals with this kernel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Convolution Results with Kernel {kernel_name}: {kernel}', fontsize=14)
    
    for ax, (name, signal) in zip(axes.flatten(), noisy_signals.items()):
        convolved = convolve(signal, kernel)
        n_conv = np.arange(len(convolved))
        ax.stem(n_conv, convolved)
        ax.set_title(f'{name.capitalize()} + Kernel {kernel_name}')
        ax.set_xlabel('n')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'outputs/kernel_{kernel_name}/comparison_kernel_{kernel_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ==================== SECTION 5: Comparison Plot ====================

# Create a comprehensive comparison plot showing all kernels applied to the sinusoidal signal
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Comparison of All Kernels Applied to Sinusoidal Signal', fontsize=14)

for ax, (kernel_name, kernel) in zip(axes.flatten(), kernels.items()):
    convolved = convolve(noisy_signals['sinusoidal'], kernel)
    n_conv = np.arange(len(convolved))
    ax.stem(n_conv, convolved)
    ax.set_title(f'Kernel {kernel_name}: {kernel}')
    ax.set_xlabel('n')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

plt.tight_layout()
plt.savefig('outputs/5_comparison_all_kernels_sinusoidal.png', dpi=150, bbox_inches='tight')
plt.close()

print("All visualizations saved to outputs/ folder!")
print("\nGenerated files:")
for root, dirs, files in os.walk('outputs'):
    for file in sorted(files):
        print(f"  {os.path.join(root, file)}")
