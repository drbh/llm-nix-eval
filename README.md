# nix-wasm-inference

> for those rare cases where you want to run an llm inside of a nix eval 😆

this repo runs llm inference inside the nix evaluator. no derivations or builds, just `nix eval`. a small rust module built on `candle-core` is compiled to wasm and called through the new `builtins.wasm` in determinate nix https://determinate.systems/blog/builtins-wasm/

the model is `Qwen/Qwen2.5-0.5B-Instruct` (the `q4_k_m` gguf, ~491MB). it needs determinate nix >= 3.17 and `--extra-experimental-features wasm-builtin`

example

```bash
nix eval --extra-experimental-features wasm-builtin --json --file example.nix
```

outputs

```
{"categories":["library","application","font"],"haiku":"Builds are reproducible,\nEach step is meticulously planned,\nSuccess in every build."}
```

or through the flake

```bash
nix eval --extra-experimental-features wasm-builtin .#lib \
  --apply 'q: q.ask "What is Nix in one sentence?"'
```

outputs

```
"Nix is a package manager for Linux that simplifies the management of software packages, allowing users to install, update, and remove packages using a single command, making it easier to manage complex software environments."
```


`generate` takes an attrset and returns `{ text; finishReason; promptTokens; completionTokens; tokenIds; }`

```nix
(import ./qwen.nix { }).generate {
  prompt = "...";        # required
  system = "...";        # defaults to the qwen system prompt
  raw = false;           # true skips the chat template
  maxTokens = 64;
  temperature = null;    # null is greedy
  topP = null;
  seed = 42;
  stream = false;        # true emits each token as a nix warning
}
```

`classify` takes the same args plus `choices` and returns `{ choice; scores; }`. it scores each choice as the reply instead of generating, so the answer is always one of the choices (the 0.5b model drifts off a label set when it generates freely)

```nix
(import ./qwen.nix { }).classify {
  prompt = "firefox";
  choices = [ "library" "application" "font" ];
}
```

`qwen.nix` grabs the weights and `tokenizer.json` with `builtins.fetchurl` (pinned hashes so it works in pure eval) and passes them in as nix paths. the wasm side reads them with the host `read_file_v2`, loads `quantized_qwen2` from `candle-transformers`, tokenizes with `tokenizers` (`unstable_wasm`) and runs the whole generate loop in a single call. sampling is seeded so the same args always give the same text

`qwen_wasm.wasm` is checked in so eval never needs IFD, same as the blog post suggests. `nix build` reproduces it

load + prefill is ~4.5s and decode is ~9 tok/s on an m-series mac (single threaded wasmtime with the simd128 quantized kernels). nix makes a fresh instance per call so the model is reloaded every time
