with import <nixpkgs> {};

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.numpy
    python3Packages.torchWithRocm
    python3Packages.torchvision
    python3Packages.freetype-py
    python3Packages.pillow
    nodePackages.pyright
    imagemagick
    ruff
    black
    python3
    nodePackages.vscode-langservers-extracted
    nodePackages.prettier
    htmlhint
    nodePackages.eslint
    stylelint
  ];

  NODE_PATH = "${pkgs.nodePackages.eslint}/lib/node_modules";
  HSA_OVERRIDE_GFX_VERSION = "10.3.0";
}

