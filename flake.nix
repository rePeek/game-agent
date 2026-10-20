{
  description = "game-agent — GCC C++23 development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell.override { stdenv = pkgs.stdenv; } {
            packages = with pkgs; [
              # Build system: CMake presets drive Ninja.
              cmake
              ninja
              pkg-config

              # Compilation cache.
              ccache

              # clang-format, clang-tidy and clangd are tools, not the compiler.
              # Pinned to the same major as the CI runner (Ubuntu's clang-format
              # 18), so formatting is deterministic across both.
              llvmPackages_18.clang-tools

              # Resolved through find_package(OpenSSL).
              openssl

              git
              gdb
            ];

            # find_package(OpenSSL): the headers and the .so symlinks live in
            # the split "dev" output of the nixpkgs package.
            OPENSSL_ROOT_DIR = "${pkgs.openssl.dev}";

            shellHook = ''
              echo ""
              echo "  game-agent dev shell"
              echo "  compiler: $(g++ --version | head -n 1)"
              echo ""
              echo "  Configure: cmake --preset debug"
              echo "  Build:     cmake --build --preset debug --parallel"
              echo "  Test:      ctest --preset debug"
              echo "  Format:    cmake --build --preset debug --target format"
              echo ""
            '';
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
