{ pkgs ? import <nixpkgs> { 
    config.allowUnfree = true; 
    config.cudaSupport = true; 
  } 
}:
let
  cuda-toolkit = pkgs.cudaPackages.cuda_nvcc;
  cuda-libs = with pkgs.cudaPackages; [
    cuda_cudart
    libcublas
    libcufft
    libcurand
    nccl
    cudnn
  ];
in

pkgs.mkShell {
  name = "cuda-env";
  buildInputs = [
    cuda-toolkit
    (pkgs.python3.withPackages (ps: with ps; [
      albumentations
      jupyterlab
      torch-bin
      torchvision-bin
      torchaudio-bin
      seaborn
    ]))
  ] ++ cuda-libs;

  shellHook = ''
    export CUDA_PATH=${cuda-toolkit}
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath cuda-libs}:$LD_LIBRARY_PATH"
    
    # Extra hint for PyTorch/XLA and other libraries
    export EXTRA_LDFLAGS="-L/run/opengl-driver/lib"
    export EXTRA_CCFLAGS="-I/run/opengl-driver/include"

    echo "--- CUDA Development Environment ---"
    # The '|| echo' prevents the shell from crashing if the GPU is fully suspended
    echo "GPU Status: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'GPU Suspended')"
    echo "NVCC Version: $(nvcc --version | grep release)"
    echo "Python: $(python --version)"
  '';
}
