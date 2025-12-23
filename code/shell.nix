{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.virtualenv  # Optional, but good to have
    pkgs.python3Packages.rasterio  # Nix-built rasterio with GDAL
    pkgs.python3Packages.pillow    # For PIL/Image
    pkgs.gcc.cc.lib  # This provides libstdc++.so.6
  ];

  shellHook = ''
    # Create a virtualenv if it doesn't exist
    if [ ! -d ".venv" ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip
      pip install earthengine-api geemap 
    else
      source .venv/bin/activate
    fi

    # Make libstdc++.so.6 (and other GCC libs) available
    export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
