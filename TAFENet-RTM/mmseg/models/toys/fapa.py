import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, Softmax, Parameter, Linear, BatchNorm2d, LayerNorm, Dropout, BatchNorm1d, Conv1d
import math
import traceback # For printing detailed error messages

# --- Placeholder weight_init ---
# Replace this with your actual weight initialization logic if needed
def weight_init(m):
    # print(f"Initializing: {type(m)}") # Optional: See which modules are initialized
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        # Basic initialization example
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# --- Module Definitions (Copied from previous response) ---

class DAP(nn.Module):
    def __init__(self, input_c=64, kernel_size=2, stride=2):
        super(DAP, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1x1 = nn.Conv2d(input_c, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        weights = self.sigmoid(self.bn(self.conv1x1(x)))
        _, _, H_w, W_w = weights.size()
        # Allow for slight floating point inaccuracies in H/W calculation if needed
        # assert H == H_w and W == W_w, f"Input size H={H},W={W} and weights size H_w={H_w},W_w={W_w} must be the same."

        # Calculate output dimensions carefully
        # Formula: floor(((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)
        # Here padding=0, dilation=1
        H1 = (H - self.kernel_size) // self.stride + 1
        W1 = (W - self.kernel_size) // self.stride + 1

        # Check if unfold is possible
        if H < self.kernel_size or W < self.kernel_size:
             print(f"Warning: Input size ({H}x{W}) is smaller than kernel size ({self.kernel_size}) in DAP. Returning zero tensor.")
             # Decide how to handle this: return zeros, average pool, etc.
             # Returning zeros of the calculated output shape
             return torch.zeros((B, C, H1, W1), device=x.device, dtype=x.dtype)


        x_unfolded = F.unfold(x, self.kernel_size, stride=self.stride)
        weights_unfolded = F.unfold(weights, self.kernel_size, stride=self.stride)

        x_unfolded = x_unfolded.view(B, C, self.kernel_size * self.kernel_size, H1 * W1)
        weights_unfolded = weights_unfolded.view(B, 1, self.kernel_size * self.kernel_size, H1 * W1)

        weighted_feature = x_unfolded * weights_unfolded
        output = weighted_feature.sum(dim=2)
        output = output.view(B, C, H1, W1)
        return output

    def initialize(self):
        weight_init(self)


class FAPAEnc_v2(Module):
    def __init__(self, in_channels, ksize):
        super(FAPAEnc_v2, self).__init__()
        # Ensure ksize divisions result in integers >= 1
        ksize_2 = max(1, ksize // 2)
        ksize_3 = max(1, ksize // 3)
        ksize_6 = max(1, ksize // 6)

        # print(f"FAPAEnc ksizes: {ksize}, {ksize_2}, {ksize_3}, {ksize_6}") # Debug ksizes

        self.pool1 = DAP(in_channels, ksize, ksize)
        self.pool2 = DAP(in_channels, ksize_2, ksize_2)
        self.pool3 = DAP(in_channels, ksize_3, ksize_3)
        self.pool4 = DAP(in_channels, ksize_6, ksize_6)

        self.conv1 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                ReLU(True))

        self.conv_fuse = Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # print(f"[FAPAEnc_v2] Input shape: {x.shape}") # Debug

        # --- DAP Pooling ---
        # Handle potential size issues where input H/W < kernel_size
        pooled1 = self.pool1(x)
        pooled2 = self.pool2(x)
        pooled3 = self.pool3(x)
        pooled4 = self.pool4(x)

        # print(f"[FAPAEnc_v2] Pooled shapes: {pooled1.shape}, {pooled2.shape}, {pooled3.shape}, {pooled4.shape}") # Debug

        # --- Convolutions and Flattening ---
        feat1 = self.conv1(pooled1).view(b, c, -1)
        feat2 = self.conv2(pooled2).view(b, c, -1)
        feat3 = self.conv3(pooled3).view(b, c, -1)
        feat4 = self.conv4(pooled4).view(b, c, -1)

        # print(f"[FAPAEnc_v2] Flattened feat shapes: {feat1.shape}, {feat2.shape}, {feat3.shape}, {feat4.shape}") # Debug

        # --- Concatenation and Fusion ---
        y_concat = torch.cat((feat1, feat2, feat3, feat4), 2)
        # print(f"[FAPAEnc_v2] Concatenated y shape: {y_concat.shape}") # Debug

        if y_concat.shape[2] == 0: # Handle case where all pooling results are empty
             print(f"Warning: FAPAEnc_v2 produced empty concatenated features for input size {h}x{w}. Returning zeros.")
             # Need to determine a reasonable K_total substitute or handle upstream
             # Returning zeros with K_total=1 as a placeholder, might need adjustment
             return torch.zeros((b, c, 1), device=x.device, dtype=x.dtype)


        y_fused = self.conv_fuse(y_concat)
        # print(f"[FAPAEnc_v2] Fused y shape: {y_fused.shape}") # Debug

        return y_fused

    def initialize(self):
        weight_init(self)


class FAPA_v2(Module):
    def __init__(self, in_channels, ksize, num_heads=8, dropout_rate=0.1, debug=False): # Added debug flag
        super(FAPA_v2, self).__init__()
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale_factor = self.head_dim ** -0.5
        self.debug = debug # Store debug flag

        self.FAPAEnc = FAPAEnc_v2(in_channels, ksize)

        self.norm1 = LayerNorm(in_channels)
        self.norm_y = LayerNorm(in_channels)
        self.norm2 = LayerNorm(in_channels)
        self.dropout = Dropout(dropout_rate) # Note: dropout is active during eval if not model.eval()

        self.query_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.softmax = Softmax(dim=-1)

        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_dropout = Dropout(dropout_rate)

        self.scale = Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        if self.debug: print(f"\n--- [FAPA_v2 Debug] Input x shape: {x.shape} ---")

        # --- Pre-Normalization ---
        # LayerNorm expects (B, ..., C)
        x_permuted = x.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        x_norm_permuted = self.norm1(x_permuted)
        x_norm = x_norm_permuted.permute(0, 2, 1).view(B, C, H, W) # (B, C, H, W)
        if self.debug: print(f"[FAPA_v2 Debug] x_norm shape: {x_norm.shape}")

        # --- Context Features ---
        y = self.FAPAEnc(x) # (B, C, K_total)
        if self.debug: print(f"[FAPA_v2 Debug] Context y shape: {y.shape}")

        # Check for empty context features
        if y.shape[2] == 0:
            print(f"Warning: FAPAEnc returned empty features. Skipping attention and returning input.")
            return x # Or handle differently, e.g., return normalized input

        # --- Context Normalization ---
        # LayerNorm expects (B, ..., C)
        y_permuted = y.permute(0, 2, 1) # (B, K_total, C)
        y_norm_permuted = self.norm_y(y_permuted)
        y_norm = y_norm_permuted.permute(0, 2, 1) # (B, C, K_total)
        if self.debug: print(f"[FAPA_v2 Debug] y_norm shape: {y_norm.shape}")

        # --- Q, K, V Generation ---
        q = self.query_conv(x_norm).view(B, C, N) # (B, C, N)
        k = self.key_conv(y_norm) # (B, C, K_total)
        v = self.value_conv(y_norm) # (B, C, K_total)
        if self.debug: print(f"[FAPA_v2 Debug] Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # --- Multi-Head Reshape ---
        # Reshape Q: (B, C, N) -> (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.permute(0, 2, 1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Reshape K: (B, C, K_total) -> (B, K_total, C) -> (B, K_total, num_heads, head_dim) -> (B, num_heads, K_total, head_dim)
        k = k.permute(0, 2, 1).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_total = k.size(2) # Get K_total dimension after reshape
        # Reshape V: (B, C, K_total) -> (B, K_total, C) -> (B, K_total, num_heads, head_dim) -> (B, num_heads, K_total, head_dim)
        v = v.permute(0, 2, 1).reshape(B, K_total, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Combine Batch and Head dims for bmm: (B * num_heads, SeqLen, head_dim)
        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, K_total, self.head_dim)
        v = v.reshape(B * self.num_heads, K_total, self.head_dim)
        if self.debug: print(f"[FAPA_v2 Debug] Multi-head Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # --- Attention Calculation ---
        # (B*H, N, C/H) @ (B*H, C/H, K) -> (B*H, N, K)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale_factor
        if self.debug: print(f"[FAPA_v2 Debug] Attention scores shape: {attn_scores.shape}")

        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropout(attn_probs) # Apply dropout to attention weights
        if self.debug: print(f"[FAPA_v2 Debug] Attention probs shape: {attn_probs.shape}")

        # (B*H, N, K) @ (B*H, K, C/H) -> (B*H, N, C/H)
        attn_output = torch.bmm(attn_probs, v)
        if self.debug: print(f"[FAPA_v2 Debug] Attention output (bmm) shape: {attn_output.shape}")

        # --- Output Processing ---
        # Reshape back: (B*H, N, C/H) -> (B, H, N, C/H) -> (B, N, H, C/H) -> (B, N, C)
        attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
        if self.debug: print(f"[FAPA_v2 Debug] Attention output reshaped: {attn_output.shape}")

        attn_output = self.proj(attn_output)
        attn_output = self.proj_dropout(attn_output) # Apply dropout after projection
        if self.debug: print(f"[FAPA_v2 Debug] Attention output after proj: {attn_output.shape}")

        # Reshape to image format: (B, N, C) -> (B, C, N) -> (B, C, H, W)
        attn_output_img = attn_output.permute(0, 2, 1).view(B, C, H, W)
        if self.debug: print(f"[FAPA_v2 Debug] Attention output image shape: {attn_output_img.shape}")

        # --- Residual Connection & Post-Normalization ---
        out = x + self.scale * attn_output_img
        if self.debug: print(f"[FAPA_v2 Debug] Output after residual: {out.shape}")

        # Apply final LayerNorm
        out_permuted = out.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        out_norm_permuted = self.norm2(out_permuted)
        out_final = out_norm_permuted.permute(0, 2, 1).view(B, C, H, W) # (B, C, H, W)
        if self.debug: print(f"[FAPA_v2 Debug] Final output shape: {out_final.shape}")
        if self.debug: print(f"--- [FAPA_v2 Debug] End ---")

        return out_final

    def initialize(self):
        weight_init(self)

# --- Main Debug Execution ---
if __name__ == "__main__":
    print("--- Starting Debug Script ---")

    # --- Hyperparameters ---
    batch_size = 2
    in_channels = 64 # Must be divisible by num_heads
    ksize = 8       # Base kernel size for DAP in FAPAEnc
    num_heads = 8
    dropout_rate = 0.0 # Set to 0 for deterministic debugging of shapes
    H, W = 55, 55    # Example input height and width

    # --- Create Dummy Input ---
    # Use a specific seed for reproducibility if needed
    # torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, in_channels, H, W)
    print(f"Input Tensor Shape: {dummy_input.shape}")
    print(f"Device: {dummy_input.device}")


    # --- Instantiate Module ---
    # Set debug=True to enable internal print statements
    try:
        fapa_module = FAPA_v2(in_channels=in_channels,
                              ksize=ksize,
                              num_heads=num_heads,
                              dropout_rate=dropout_rate,
                              debug=True) # Enable debug prints

        # Initialize weights (optional but good practice)
        fapa_module.initialize()
        print(f"Successfully instantiated FAPA_v2 module.")

        # Set to evaluation mode if dropout should be disabled
        # fapa_module.eval()

        # --- Forward Pass ---
        print("\n--- Performing Forward Pass ---")
        try:
            # Ensure module and data are on the same device (if using GPU)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # fapa_module.to(device)
            # dummy_input = dummy_input.to(device)

            with torch.no_grad(): # Disable gradient calculation for debugging
                output = fapa_module(dummy_input)

            print("\n--- Forward Pass Successful ---")
            print(f"Output Tensor Shape: {output.shape}")

            # --- Basic Checks ---
            assert output.shape == dummy_input.shape, \
                f"Output shape {output.shape} does not match input shape {dummy_input.shape}!"
            print("Check PASSED: Output shape matches input shape.")

            # Check for NaNs/Infs
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("Warning: Output contains NaN or Inf values!")
            else:
                print("Check PASSED: Output does not contain NaN or Inf.")


        except Exception as e:
            print("\n--- Error during Forward Pass ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print("Traceback:")
            traceback.print_exc() # Print detailed traceback

    except Exception as e:
        print("\n--- Error during Module Instantiation or Initialization ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc()

    print("\n--- Debug Script Finished ---")