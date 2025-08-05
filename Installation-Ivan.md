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