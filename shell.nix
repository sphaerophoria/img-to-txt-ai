with import <nixpkgs> {};

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.numpy
    python3Packages.torchWithRocm
    python3Packages.freetype-py
    python3Packages.pillow
    nodePackages.pyright
    ruff
    black
  ];
}

