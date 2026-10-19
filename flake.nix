{
  description = "game-agent — a structured tool-calling C++ SDK for OpenAI-compatible models";

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
      #   * openssl — the HTTP transport links it (zlib was dropped with the
      #     streaming support, and clickhouse-cpp with its integration test).
      #   * nlohmann/json and GoogleTest are resolved or fetched by CMake
      #     during configure, hence the need for network access in the shell —
      #     that is expected and works fine outside the Nix sandbox.
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
                echo "  game-agent dev shell · ${name}"
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
            name = "game-agent";
            stdenv = pkgs.stdenv;
          };

          # Same environment, but built with the nixpkgs Clang toolchain.
          clang = mkDevShell {
            name = "game-agent-clang";
            stdenv = pkgs.llvmPackages.stdenv;
          };
        };
    in
    {
      devShells = forAllSystems devShellsFor;

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
