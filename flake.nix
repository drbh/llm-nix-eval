{
  description = "Qwen2.5 inference inside the Nix evaluator via builtins.wasm + candle";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1";

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # Evaluation-time API. Uses the committed qwen_wasm.wasm so that calling
      # it never needs import-from-derivation; `nix build` reproduces that file.
      lib = import ./qwen.nix { };

      # nix eval --extra-experimental-features wasm-builtin --json .#example
      example = import ./example.nix;

      packages = forAllSystems (pkgs: {
        default = pkgs.rustPlatform.buildRustPackage {
          pname = "qwen-wasm";
          version = "0.1.0";
          src = pkgs.lib.cleanSource ./.;
          cargoHash = "sha256-2WF2XoIt48TcmRv6gp3OYpzbFbMcntHCWA9y+/BLSWA=";

          # simd128 + custom getrandom backend come from .cargo/config.toml.
          CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
          buildPhase = "cargo build --release -p qwen-wasm";
          doCheck = false;

          nativeBuildInputs = [ pkgs.rustc.llvmPackages.lld pkgs.binaryen ];
          installPhase = ''
            mkdir -p $out
            wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
              -o $out/qwen_wasm.wasm target/wasm32-unknown-unknown/release/qwen_wasm.wasm
          '';
          allowedReferences = [ ];
        };
      });

      checks = forAllSystems (pkgs: {
        build = self.packages.${pkgs.system}.default;
      });
    };
}
