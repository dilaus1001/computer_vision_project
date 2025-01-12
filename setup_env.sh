#!/bin/bash

echo "🔍 Detecting system type..."

is_apple_silicon() {
    if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

is_linux() {
    if [[ "$(uname)" == "Linux" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

create_conda_env() {
    echo "🚀 Creating conda environment 'cvproj' with Python 3.10.16..."
    conda create -n cvproj python=3.10.16 -y
    eval "$(conda shell.bash hook)"
    conda activate cvproj
}

setup_linux() {
    echo "🐧 Setting up environment for Linux..."
    echo "📦 Installing CUDA dependencies..."
    conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9.2 -y

    echo "📦 Installing Python packages..."
    pip install -r requirements.txt

    echo "✅ Linux setup completed successfully!"
}


setup_macos() {
    echo "🍎 Setting up environment for MacOS M1..."
    
    echo "📦 Installing basic Python packages..."
    pip install numpy==1.26.3 pandas==2.2.3 matplotlib==3.10.0 seaborn==0.13.2 scipy==1.15.0
    pip install pillow==10.2.0 opencv-python==4.10.0.84 pyyaml==6.0.2 tqdm==4.67.1

    echo "📦 Installing PyTorch for M1..."
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    echo "📦 Installing additional packages..."
    pip install ultralytics==8.3.58 ultralytics-thop==2.0.13
    pip install onnx==1.17.0 onnxruntime==1.20.1
    pip install psutil==6.1.1 requests==2.32.3 filelock==3.13.1 rich==13.9.4

    echo "✅ MacOS M1 setup completed successfully!"
}

# Main script execution
echo "🎯 Starting environment setup..."

create_conda_env
if is_apple_silicon; then
    setup_macos
elif is_linux; then
    setup_linux
else
    echo "❌ Error: Unsupported system. This script only supports Linux and MacOS M1."
    exit 1
fi

echo "🎉 Setup completed! To activate the environment, run: conda activate cvproj"