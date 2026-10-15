# nix eval --extra-experimental-features wasm-builtin --file example.nix
let
  qwen = import ./qwen.nix { };
in
{
  haiku = qwen.ask "Write a haiku about reproducible builds.";
  # Use the model as an ordinary Nix function: classify package names.
  categories = map (
    pkg:
    (qwen.classify {
      system = "Classify the nixpkgs package as a library, application, or font.";
      prompt = pkg;
      choices = [ "library" "application" "font" ];
    }).choice
  ) [ "openssl" "firefox" "noto-fonts" ];
}
