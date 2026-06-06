{ pkgs ? import <nixpkgs> { 
    config = { 
      allowUnfree = true; 
      cudaSupport = true; 
    }; 
  } 
}:

let
  cuda-path = "/usr/lib/wsl/lib:/run/opengl-driver/lib"; 
in
pkgs.mkShell {
  name = "dinov3-segmentation";

  buildInputs = [
    pkgs.python311
    pkgs.python311Packages.virtualenv
    
    pkgs.cudaPackages.cudatoolkit 
    pkgs.cudaPackages.cudnn
    
    pkgs.zlib
    pkgs.libpng
    pkgs.libjpeg
    pkgs.freetype
    pkgs.stdenv.cc.cc.lib
    pkgs.git
    pkgs.gcc
    pkgs.pkg-config
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libpng
    pkgs.libjpeg
    pkgs.freetype
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cudnn
  ] + ":${cuda-path}"; 

  shellHook = ''
    VENV_DIR="$PWD/.venv"

    if [ ! -f "$VENV_DIR/bin/activate" ]; then
      echo "→ Creating virtual environment..."
      virtualenv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "→ Installing/Fixing PyTorch with CUDA 12.1..."
      pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi
      pip install --force-reinstall pandas

    export TORCH_HOME="$PWD/.torch_cache"
    export MPLBACKEND="Agg"

    echo ""
    echo "  Environment Ready"
    echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo ""
  '';
}
