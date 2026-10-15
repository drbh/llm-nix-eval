//! Qwen2.5 text generation as a Nix builtin.
//!
//! Called from Nix via `builtins.wasm { path = ...; function = "generate"; } { ... }`.
//! Everything runs inside the evaluator: weights and tokenizer are read from Nix
//! paths through the host's `read_file_v2`, and the result is a Nix attrset.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use nix_wasm_rust::{warn, Type, Value};
use tokenizers::Tokenizer;

const DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// getrandom's custom backend (selected in .cargo/config.toml). Nix offers no
/// entropy source, and the only consumer is ahash's hash-map seeding inside
/// `tokenizers`, so fixed bytes keep the builtin pure without changing results.
#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(dest: *mut u8, len: usize) -> Result<(), getrandom::Error> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *dest.add(i) = (state >> 56) as u8;
    }
    Ok(())
}

fn fail(msg: impl std::fmt::Display) -> ! {
    nix_wasm_rust::panic(&msg.to_string())
}

fn attr(args: &Value, name: &str) -> Option<Value> {
    args.get_attr(name)
}

fn required(args: &Value, name: &str) -> Value {
    attr(args, name).unwrap_or_else(|| fail(format!("qwen: missing required attribute '{name}'")))
}

fn as_f64(v: Value) -> f64 {
    match v.get_type() {
        Type::Float => v.get_float(),
        Type::Int => v.get_int() as f64,
        _ => fail("qwen: expected a number"),
    }
}

fn is_null(v: &Value) -> bool {
    matches!(v.get_type(), Type::Null)
}

struct Params {
    prompt: String,
    system: String,
    raw: bool,
    max_tokens: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: u64,
    stream: bool,
}

impl Params {
    fn from_nix(args: &Value) -> Self {
        let opt_num = |name| attr(args, name).filter(|v| !is_null(v)).map(as_f64);
        Params {
            prompt: required(args, "prompt").get_string(),
            system: attr(args, "system")
                .map(|v| v.get_string())
                .unwrap_or_else(|| DEFAULT_SYSTEM.to_string()),
            raw: attr(args, "raw").map(|v| v.get_bool()).unwrap_or(false),
            max_tokens: attr(args, "maxTokens").map(|v| v.get_int() as usize).unwrap_or(64),
            temperature: opt_num("temperature").filter(|t| *t > 0.0),
            top_p: opt_num("topP"),
            seed: attr(args, "seed").map(|v| v.get_int() as u64).unwrap_or(42),
            stream: attr(args, "stream").map(|v| v.get_bool()).unwrap_or(false),
        }
    }

    fn rendered_prompt(&self) -> String {
        if self.raw {
            return self.prompt.clone();
        }
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            self.system, self.prompt
        )
    }
}

fn load(args: &Value, device: &Device) -> (Tokenizer, ModelWeights) {
    let tokenizer_bytes = required(args, "tokenizer").read_file();
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .unwrap_or_else(|e| fail(format!("qwen: bad tokenizer: {e}")));
    drop(tokenizer_bytes);

    let gguf_bytes = required(args, "model").read_file();
    let mut reader = std::io::Cursor::new(gguf_bytes);
    let content = gguf_file::Content::read(&mut reader)
        .unwrap_or_else(|e| fail(format!("qwen: bad gguf: {e}")));
    let model = ModelWeights::from_gguf(content, &mut reader, device)
        .unwrap_or_else(|e| fail(format!("qwen: loading weights: {e}")));
    (tokenizer, model)
}

fn tokenize(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .unwrap_or_else(|e| fail(format!("qwen: tokenize: {e}")))
        .get_ids()
        .to_vec()
}

/// Runs `tokens` starting at position `pos` and returns the logits for the last one.
fn step(model: &mut ModelWeights, device: &Device, tokens: &[u32], pos: usize) -> Tensor {
    let input = Tensor::new(tokens, device)
        .and_then(|t| t.unsqueeze(0))
        .unwrap_or_else(|e| fail(e));
    model
        .forward(&input, pos)
        .and_then(|l| l.squeeze(0))
        .unwrap_or_else(|e| fail(format!("qwen: forward: {e}")))
}

fn log_softmax(logits: &Tensor) -> Vec<f32> {
    candle_nn::ops::log_softmax(logits, 0)
        .and_then(|l| l.to_vec1())
        .unwrap_or_else(|e| fail(e))
}

#[no_mangle]
pub extern "C" fn generate(args: Value) -> Value {
    let params = Params::from_nix(&args);
    let device = Device::Cpu;

    let (tokenizer, mut model) = load(&args, &device);

    let prompt_tokens = tokenize(&tokenizer, &params.rendered_prompt());

    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    let sampling = match params.temperature {
        None => Sampling::ArgMax,
        Some(temperature) => match params.top_p {
            Some(p) => Sampling::TopP { p, temperature },
            None => Sampling::All { temperature },
        },
    };
    let mut logits_processor = LogitsProcessor::from_sampling(params.seed, sampling);

    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens);
    let mut logits = step(&mut model, &device, &prompt_tokens, 0);
    let mut finish_reason = "length";
    while generated.len() < params.max_tokens {
        let next = logits_processor
            .sample(&logits)
            .unwrap_or_else(|e| fail(format!("qwen: sample: {e}")));
        if eos.contains(&next) {
            finish_reason = "stop";
            break;
        }
        generated.push(next);
        if params.stream {
            let piece = tokenizer.decode(&[next], true).unwrap_or_default();
            warn!("[{}/{}] {:?}", generated.len(), params.max_tokens, piece);
        }
        logits = step(&mut model, &device, &[next], prompt_tokens.len() + generated.len() - 1);
    }

    let text = tokenizer
        .decode(&generated, true)
        .unwrap_or_else(|e| fail(format!("qwen: detokenize: {e}")));
    let token_ids: Vec<Value> = generated.iter().map(|&t| Value::make_int(t as i64)).collect();

    Value::make_attrset(&[
        ("text", Value::make_string(&text)),
        ("finishReason", Value::make_string(finish_reason)),
        ("promptTokens", Value::make_int(prompt_tokens.len() as i64)),
        ("completionTokens", Value::make_int(generated.len() as i64)),
        ("tokenIds", Value::make_list(&token_ids)),
    ])
}

/// Picks whichever of `choices` the model finds most likely as the reply, by
/// mean token log-probability. Small models drift off a label set when asked to
/// generate freely; scoring the labels keeps the answer inside it.
#[no_mangle]
pub extern "C" fn classify(args: Value) -> Value {
    let params = Params::from_nix(&args);
    let choices: Vec<String> = required(&args, "choices")
        .get_list()
        .iter()
        .map(|v| v.get_string())
        .collect();
    if choices.is_empty() {
        fail("qwen: 'choices' must not be empty");
    }
    let device = Device::Cpu;
    let (tokenizer, mut model) = load(&args, &device);
    let prompt_tokens = tokenize(&tokenizer, &params.rendered_prompt());

    let scores: Vec<f64> = choices
        .iter()
        .map(|choice| {
            let choice_tokens = tokenize(&tokenizer, choice);
            model.clear_kv_cache();
            let mut logits = step(&mut model, &device, &prompt_tokens, 0);
            let mut total = 0f64;
            for (i, &tok) in choice_tokens.iter().enumerate() {
                total += log_softmax(&logits)[tok as usize] as f64;
                if i + 1 < choice_tokens.len() {
                    logits = step(&mut model, &device, &[tok], prompt_tokens.len() + i);
                }
            }
            total / choice_tokens.len().max(1) as f64
        })
        .collect();

    let best = (0..choices.len())
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .unwrap();
    let score_attrs: Vec<(&str, Value)> = choices
        .iter()
        .zip(&scores)
        .map(|(c, &s)| (c.as_str(), Value::make_float(s)))
        .collect();

    Value::make_attrset(&[
        ("choice", Value::make_string(&choices[best])),
        ("scores", Value::make_attrset(&score_attrs)),
    ])
}
