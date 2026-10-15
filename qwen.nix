# A tiny LLM inference engine that runs entirely inside the Nix evaluator.
#
#   let qwen = import ./qwen.nix { }; in qwen.generate { prompt = "Hi!"; }
#
# Requires Determinate Nix >= 3.17 with `--extra-experimental-features wasm-builtin`.
{
  wasm ? ./qwen_wasm.wasm,
}:
let
  # builtins.fetchurl returns a store-path *string*; the Wasm host's
  # read_file_v2 wants a Nix *path* value, so turn it back into one.
  fetchPath = args: /. + builtins.unsafeDiscardStringContext (builtins.fetchurl args);

  model = fetchPath {
    url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    sha256 = "sha256-dKTajJ/bzRW9H20B1iFBDTHG/ACYb162h4JOe5PXqds=";
  };

  tokenizer = fetchPath {
    url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
    sha256 = "sha256-wDghF+oynN8JcEETL21zWSS2l5JNb2/DlFcT6WzodTk=";
  };

  call =
    function:
    builtins.wasm {
      path = wasm;
      inherit function;
    };
in
rec {
  inherit model tokenizer;

  # Full result: { text, finishReason, promptTokens, completionTokens, tokenIds }.
  # Options: prompt (required), system, raw, maxTokens (64), temperature
  # (null = greedy), topP, seed (42), stream (emit partial text as warnings).
  generate = opts: call "generate" ({ inherit model tokenizer; } // opts);

  # Scores each of `choices` as the reply and returns { choice, scores }.
  # Takes the same prompt options as generate, plus choices (list of strings).
  classify = opts: call "classify" ({ inherit model tokenizer; } // opts);

  # Just the text.
  ask = prompt: (generate { inherit prompt; }).text;
}
