{
  description = "ai-sdk-cpp — a unified C++ SDK for multiple AI/LLM providers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;

      # Everything a developer needs regardless of the compiler in use.
      #
      # Notes on the dependency list:
      #   * cmake/ninja/pkg-config — `scripts/build.py` always configures with
      #     `-G Ninja`, so Ninja is not optional.
      #   * clang-tools — provides clang-format, clang-tidy and clangd for
      #     `scripts/format.py`, `scripts/lint.py` and editor integration.
      #   * uv + python3 — the scripts under scripts/ are PEP 723 scripts run
      #     with `uv run`.
      #   * openssl/zlib — the project links OpenSSL and zlib. zlib itself is
      #     also fetched by CMake (third_party/zlib-cmake), but having it in the
      #     shell keeps `find_package(ZLIB)` happy for third-party code.
      #   * Zstd/LZ4/cityhash/abseil are *not* listed: clickhouse-cpp (test-only
      #     dependency) vendors them by default (WITH_SYSTEM_* = OFF).
      #   * GoogleTest and clickhouse-cpp are downloaded by CMake's
      #     FetchContent during configure, hence the need for network access in
      #     the shell — that is expected and works fine outside the Nix sandbox.
      commonInputs =
        pkgs:
        with pkgs;
        [
          # Build system
          cmake
          ninja
          pkg-config

          # Compilation cache
          ccache

          # C++ tooling: clang-format, clang-tidy, clangd
          clang-tools

          # Libraries resolved via find_package()
          openssl
          zlib

          # Script runner and interpreter for scripts/*.py
          uv
          python3

          # Misc
          git
        ]
        ++ lib.optionals stdenv.hostPlatform.isLinux [ gdb ]
        ++ lib.optionals stdenv.hostPlatform.isDarwin [ lldb ];

      devShellsFor =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          inherit (pkgs) lib;

          mkDevShell =
            {
              name,
              stdenv,
            }:
            pkgs.mkShell.override { inherit stdenv; } {
              inherit name;

              packages = commonInputs pkgs;

              env = {
                # find_package(OpenSSL): the headers and the .so symlinks live in
                # the split "dev" output of the nixpkgs package.
                OPENSSL_ROOT_DIR = "${pkgs.openssl.dev}";

                # clangd picks up build/compile_commands.json automatically once
                # CMake has been run once.
                CMAKE_EXPORT_COMPILE_COMMANDS = "ON";

                # Never let uv download its own (non-NixOS) CPython builds; use
                # the python3 shipped in this shell instead.
                UV_PYTHON_DOWNLOADS = "never";
              };

              shellHook = ''
                echo ""
                echo "  ai-sdk-cpp dev shell · ${name}"
                echo "  compiler: $(${stdenv.cc.targetPrefix}c++ --version | head -n 1)"
                echo ""
                echo "  Build:  uv run scripts/build.py --tests --export-compile-commands"
                echo "  Test:   (cd build && ctest --build-config Debug --output-on-failure)"
                echo "  Format: uv run scripts/format.py --check"
                echo "  Lint:   uv run scripts/lint.py"
                echo ""
              '';
            };
        in
        {
          # Default shell: nixpkgs' GCC toolchain.
          default = mkDevShell {
            name = "ai-sdk-cpp";
            stdenv = pkgs.stdenv;
          };

          # Same environment, but built with the nixpkgs Clang toolchain.
          clang = mkDevShell {
            name = "ai-sdk-cpp-clang";
            stdenv = pkgs.llvmPackages.stdenv;
          };
        };
    in
    {
      devShells = forAllSystems devShellsFor;

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
