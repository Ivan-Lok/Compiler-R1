# Installation Guide

## Prerequisites
Before you begin, ensure you have the following prerequisites:
- A CUDA version 12.1 installed in your system.
    > Notes: I am not so sure if the later version would work as well, but I have tested it with 12.1.\
    > For installation of CUDA 12.1, you can refers to here [CUDA 12.1 WSL Installation](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) \
    > And Change the `sudo apt-get -y install cuda` in the last step to `sudo apt-get install cuda-compiler-12-1 cuda-libraries-12-1`
- A compatible NVIDIA GPU that supports CUDA.
    > Please check by running `nvidia-smi` in your terminal. If you see your GPU listed, you are good to go.
- Install a ninja build system.
    > You can install it using the following command:
    ```bash
    sudo apt-get install ninja-build
    ```

## Installation Steps
1. git clone the repository
2. Following the environment Setup instructions in the repository:

    ```bash
    # Create and activate conda environment
    conda create -n Compiler-R1 python==3.10
    conda activate Compiler-R1

    # Initialize and update submodules
    git submodule update --init --recursive

    # Install verl and other dependencies
    cd verl
    pip3 install -e .
    cd .. 
    pip3 install vllm
    pip3 install flash-attn --no-build-isolation
    # Noted that the flash-attn take very long time to install, please be patient.
    pip3 install FlagEmbedding
    pip3 install faiss-cpu
    ```

3. If you would like to make use of the datasets provided in the repository, you can download them using the following command:

    ```bash
    cd examples/data_preprocess/
    wget https://github.com/Mind4Compiler/Compiler-R1/releases/download/llvm_raw_code/llvmir_datasets.zip
    unzip llvmir_datasets.zip
    rm llvmir_datasets.zip
    ```
## Training Steps

### Step 1: Train the model
To train the model, you can use one of the provided training scripts:

1. **`train_Exp_1_2_Ivan.sh`** - Uses `Qwen/Qwen2.5-1.5B-Instruct` as the default model
2. **`train_Exp_1_2_DialoGPT-small.sh`** - Uses `microsoft/DialoGPT-small` as the default model for even smaller resource requirements

Please ensure you have correctly set the `CUDA_VISIBLE_DEVICES` environment variable to specify which GPU(s) to use.

**Model Options:**
- **Qwen2.5-1.5B-Instruct**: The default model in `train_Exp_1_2_Ivan.sh`, suitable for single GPU training with moderate resource requirements
- **DialoGPT-small**: An even smaller model option in `train_Exp_1_2_DialoGPT-small.sh`, ideal for systems with limited GPU memory

You can change the model by modifying the `base_model` parameter in the respective script or by using the `--model` command line argument.

## Script and Modifications

### Overview of Ivan Training Scripts

During the development and testing process, several optimized training scripts were created to address hardware compatibility issues and improve training stability. This section documents all the modifications made to create the Ivan-specific training scripts.

### Scripts Created

1. **`train_Exp_1_2_Ivan.sh`** - Single GPU optimized with HuggingFace rollout backend for maximum compatibility

### Key Modifications Made

#### 1. CUDA Architecture Compatibility
**Problem**: Original script didn't specify CUDA architecture for RTX 5880 Ada GPUs
**Solution**: Added explicit CUDA architecture support
```bash
export TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0"
```

#### 2. VLLM Custom Kernel Issues
**Problem**: VLLM custom CUDA kernels (rms_norm, attention) failed on RTX 5880 architecture
**Solution**: Disabled all problematic VLLM custom operations
```bash
export VLLM_USE_CUSTOM_LAYERNORM=0
export VLLM_USE_CUSTOM_ROTARY_EMB=0
export VLLM_USE_CUSTOM_ATTENTION=0
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_FLASH_ATTN_2_BY_DEFAULT=0
export VLLM_DISABLE_CUSTOM_OPS=1
```

#### 3. Rollout Backend Switch
**Problem**: VLLM rollout backend had persistent CUDA kernel compatibility issues
**Solution**: Switched from VLLM to HuggingFace rollout backend
```bash
# Changed from:
actor_rollout_ref.rollout.name=vllm

# To:
actor_rollout_ref.rollout.name=hf
```

#### 4. Tensor Parallelism Configuration
**Problem**: Default tensor_model_parallel_size=2 conflicted with single GPU setup (world_size=1)
**Solution**: Explicitly set tensor parallelism for single GPU
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

#### 5. FSDP Memory Management
**Problem**: Parameter offloading to CPU caused device placement errors with HuggingFace rollout
**Solution**: Adjusted FSDP configuration for better compatibility
```bash
# Actor configuration
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true

# Reference model configuration  
actor_rollout_ref.ref.fsdp_config.param_offload=false
```

#### 6. Logging Configuration
**Problem**: Weights & Biases (wandb) required API key configuration
**Solution**: Simplified to console-only logging for easier setup
```bash
# Changed from:
"trainer.logger=[console,wandb]"

# To:
"trainer.logger=[console]"
```

#### 7. Memory Optimization Settings
**Problem**: Conservative memory settings for stability on various hardware
**Solution**: Added comprehensive memory management
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
```

#### 8. Hardware-Specific Optimizations

**Single GPU Version (`train_Exp_1_2_Ivan.sh`)**:
- Conservative batch sizes (train_batch_size=16)
- CPU offloading for optimizer states
- Single tensor parallel process
- HuggingFace rollout for maximum compatibility

### Troubleshooting Solutions Implemented

#### Error Resolution Timeline:
1. **Transformers Version Incompatibility**: Fixed by disabling Ulysses sequence parallel
2. **VLLM RMS Norm Kernel Failure**: Resolved by disabling custom kernels
3. **Tensor Parallelism Mismatch**: Fixed by explicit single GPU configuration  
4. **Wandb Authentication Error**: Solved by switching to console logging
5. **FSDP Parameter Placement**: Resolved by disabling parameter offloading
6. **Hydra Configuration Syntax**: Fixed parameter override syntax issues

### Recommended Usage

**Primary Script**: Use `train_Exp_1_2_Ivan.sh`
- Most stable and compatible
- Works on single GPU setups
- Conservative memory usage
- HuggingFace rollout backend for maximum compatibility
- Optimized for RTX 5880 and similar modern GPUs

### Environment Variables Summary

The Ivan script includes these stability-focused environment variables:
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0"
```

These modifications ensure reliable training across different hardware configurations while maintaining the original functionality of the compiler optimization training pipeline.