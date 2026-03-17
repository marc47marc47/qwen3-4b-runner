use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use clap::Parser;
use hf_hub::api::sync::Api;
use rustyline::config::{CompletionType, Config as RustylineConfig, EditMode};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-4B";
const DEFAULT_QUANTIZED_MODEL_ID: &str = "Qwen/Qwen3-4B-GGUF";
const DEFAULT_GGUF_FILE: &str = "Qwen3-4B-Q4_K_M.gguf";
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful coding assistant.";

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model: String,

    #[arg(long, default_value = DEFAULT_QUANTIZED_MODEL_ID)]
    quantized_model: String,

    #[arg(long, default_value = DEFAULT_GGUF_FILE)]
    gguf_file: String,

    #[arg(long)]
    quantized: bool,

    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system: String,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    chat: bool,

    #[arg(long, default_value_t = 128)]
    sample_len: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long, default_value_t = 0.95)]
    top_p: f64,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value_t = 4096)]
    max_context: usize,

    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,

    #[arg(long)]
    cpu: bool,
}

#[derive(Debug, Deserialize)]
struct WeightsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

enum LoadedModel {
    Dense(ModelForCausalLM),
    Quantized(QuantizedModelWeights),
}

impl LoadedModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(model) => model.forward(input, index_pos),
            Self::Quantized(model) => model.forward(input, index_pos),
        }
    }

    fn forward_prompt(&mut self, input_ids: &[u32], index_pos: usize, device: &Device) -> Result<Tensor> {
        match self {
            Self::Dense(model) => {
                let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
                Ok(model.forward(&input, index_pos)?)
            }
            Self::Quantized(model) => {
                if index_pos == 0 || input_ids.len() <= 1 {
                    let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
                    return Ok(model.forward(&input, index_pos)?);
                }

                let mut logits = None;
                let mut pos = index_pos;
                for token in input_ids {
                    let input = Tensor::new(&[*token], device)?.unsqueeze(0)?;
                    logits = Some(model.forward(&input, pos)?);
                    pos += 1;
                }
                logits.context("quantized forward received empty prompt")
            }
        }
    }
}

#[derive(Clone)]
enum ModelSource {
    Dense {
        config: Config,
        weights: Vec<PathBuf>,
        dtype: DType,
    },
    Quantized {
        gguf_path: PathBuf,
    },
}

impl ModelSource {
    fn load(&self, device: &Device) -> Result<LoadedModel> {
        match self {
            Self::Dense {
                config,
                weights,
                dtype,
            } => Ok(LoadedModel::Dense(load_model(config, weights, *dtype, device)?)),
            Self::Quantized { gguf_path } => {
                Ok(LoadedModel::Quantized(load_quantized_model(gguf_path, device)?))
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.chat && args.prompt.is_none() {
        bail!("--prompt is required unless --chat is enabled");
    }

    let device = select_device(args.cpu)?;
    let dtype = preferred_dtype(&device);

    eprintln!("device: {device:?}");
    eprintln!("dtype: {dtype:?}");
    eprintln!(
        "model: {}",
        if args.quantized {
            &args.quantized_model
        } else {
            &args.model
        }
    );

    let api = Api::new().context("failed to create Hugging Face API client")?;
    let tokenizer_repo = api.model(args.model.clone());
    let tokenizer_path = tokenizer_repo
        .get("tokenizer.json")
        .context("failed to fetch tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let model_source = if args.quantized {
        let repo = api.model(args.quantized_model.clone());
        let gguf_path = repo
            .get(&args.gguf_file)
            .with_context(|| format!("failed to fetch gguf file {}", args.gguf_file))?;
        ModelSource::Quantized { gguf_path }
    } else {
        let repo = api.model(args.model.clone());
        let config_path = repo.get("config.json").context("failed to fetch config.json")?;
        let weight_paths = download_weight_files(&repo)
            .with_context(|| format!("failed to fetch weights for {}", args.model))?;
        let config = load_config(&config_path)?;
        ModelSource::Dense {
            config,
            weights: weight_paths,
            dtype,
        }
    };
    let mut model = model_source.load(&device)?;

    let eos_token = tokenizer.token_to_id("<|im_end|>");
    let mut logits_processor = LogitsProcessor::from_sampling(
        args.seed,
        Sampling::TopP {
            p: args.top_p,
            temperature: args.temperature,
        },
    );

    if args.chat {
        run_chat_loop(
            &args,
            &tokenizer,
            &mut model,
            &model_source,
            &mut logits_processor,
            eos_token,
            &device,
        )?;
    } else {
        let prompt = build_initial_prompt(
            &args.system,
            args.prompt.as_deref().expect("validated above"),
        );
        let mut state = ChatState::new();
        let input_ids = encode_prompt(&tokenizer, &prompt)?;
        if input_ids.is_empty() {
            bail!("prompt produced no tokens");
        }
        let generated = generate_reply(
            &args,
            &tokenizer,
            &mut model,
            &model_source,
            &mut logits_processor,
            &mut state,
            &input_ids,
            eos_token,
            &device,
            false,
        )?;
        println!("{generated}");
    }
    Ok(())
}

#[derive(Default)]
struct ChatState {
    turns: Vec<(String, String)>,
    tokens: Vec<u32>,
}

impl ChatState {
    fn new() -> Self {
        Self::default()
    }
}

fn select_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(0).context("failed to initialize CUDA device 0");
    }

    #[cfg(not(feature = "cuda"))]
    {
        Ok(Device::Cpu)
    }
}

fn preferred_dtype(device: &Device) -> DType {
    match device {
        Device::Cpu => DType::F32,
        _ => DType::BF16,
    }
}

fn build_initial_prompt(system: &str, user_prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn build_turn_prompt(user_prompt: &str) -> String {
    format!("<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer.encode(prompt, false).map_err(anyhow::Error::msg)?;
    Ok(encoding.get_ids().to_vec())
}

fn run_chat_loop(
    args: &Args,
    tokenizer: &Tokenizer,
    model: &mut LoadedModel,
    model_source: &ModelSource,
    logits_processor: &mut LogitsProcessor,
    eos_token: Option<u32>,
    device: &Device,
) -> Result<()> {
    let mut state = ChatState::new();
    let config = RustylineConfig::builder()
        .edit_mode(EditMode::Vi)
        .completion_type(CompletionType::List)
        .build();
    let mut editor =
        DefaultEditor::with_config(config).context("failed to initialize line editor")?;

    println!("chat mode: type 'exit' or 'quit' to leave");

    loop {
        let line = match editor.readline("you> ") {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                println!();
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(err) => return Err(err).context("failed to read from line editor"),
        };
        let user_input = line.trim();
        if user_input.is_empty() {
            continue;
        }
        if matches!(user_input, "exit" | "quit") {
            break;
        }
        editor
            .add_history_entry(user_input)
            .context("failed to record input history")?;

        let prompt = if state.turns.is_empty() && state.tokens.is_empty() {
            build_initial_prompt(&args.system, user_input)
        } else {
            build_turn_prompt(user_input)
        };

        let input_ids = encode_prompt(tokenizer, &prompt)?;
        print!("assistant> ");
        io::stdout().flush().context("failed to flush stdout")?;
        let generated = generate_reply(
            args,
            tokenizer,
            model,
            model_source,
            logits_processor,
            &mut state,
            &input_ids,
            eos_token,
            device,
            true,
        )?;
        state.turns.push((user_input.to_string(), generated.clone()));
        if generated.is_empty() {
            println!();
        }
    }

    Ok(())
}

fn generate_reply(
    args: &Args,
    tokenizer: &Tokenizer,
    model: &mut LoadedModel,
    model_source: &ModelSource,
    logits_processor: &mut LogitsProcessor,
    state: &mut ChatState,
    input_ids: &[u32],
    eos_token: Option<u32>,
    device: &Device,
    stream: bool,
) -> Result<String> {
    if input_ids.is_empty() {
        bail!("prompt produced no tokens");
    }

    if state.tokens.len() + input_ids.len() + args.sample_len > args.max_context {
        rebuild_state_for_current_turn(args, tokenizer, model, model_source, state, input_ids, device)?;
    }

    if state.tokens.len() + input_ids.len() > args.max_context {
        bail!(
            "prompt exceeds --max-context ({} tokens). Reduce prompt size or raise the limit.",
            args.max_context
        );
    }

    eprintln!("prompt_tokens: {}", input_ids.len());
    eprintln!("sampling {} new tokens...", args.sample_len);

    let logits = model.forward_prompt(input_ids, state.tokens.len(), device)?;
    let mut logits = logits.squeeze(0)?.squeeze(0)?;
    state.tokens.extend_from_slice(input_ids);

    let generated_start = state.tokens.len();
    let mut visible_text = String::new();
    let mut streamed_tokens = Vec::new();

    for _ in 0..args.sample_len {
        let logits_with_penalty = if args.repeat_penalty > 1.0 {
            let start_at = state.tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&logits, args.repeat_penalty, &state.tokens[start_at..])?
        } else {
            logits.clone()
        };

        let next_token = logits_processor.sample(&logits_with_penalty)?;
        state.tokens.push(next_token);
        if eos_token.is_some_and(|id| id == next_token) {
            break;
        }

        if stream {
            streamed_tokens.push(next_token);
            let decoded = tokenizer
                .decode(&streamed_tokens, true)
                .map_err(anyhow::Error::msg)?;
            let stable = stable_prefix(&decoded);
            if let Some(suffix) = stable.strip_prefix(&visible_text) {
                if !suffix.is_empty() {
                    print!("{suffix}");
                    io::stdout().flush().context("failed to flush stdout")?;
                    visible_text.push_str(suffix);
                }
            }
        }

        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        logits = model.forward(&input, state.tokens.len() - 1)?;
        logits = logits.squeeze(0)?.squeeze(0)?;
    }

    let generated_tokens = match eos_token {
        Some(eos) => {
            let end = state
                .tokens
                .iter()
                .rposition(|&token| token != eos)
                .map(|idx| idx + 1)
                .unwrap_or(generated_start);
            &state.tokens[generated_start..end]
        }
        None => &state.tokens[generated_start..],
    };

    let generated = if stream {
        let final_text = tokenizer
            .decode(generated_tokens, true)
            .map_err(anyhow::Error::msg)?;
        if let Some(suffix) = final_text.strip_prefix(&visible_text) {
            if !suffix.is_empty() {
                print!("{suffix}");
                io::stdout().flush().context("failed to flush stdout")?;
                visible_text.push_str(suffix);
            }
        }
        println!();
        visible_text
    } else {
        tokenizer
            .decode(generated_tokens, true)
            .map_err(anyhow::Error::msg)?
    };
    Ok(generated)
}

fn stable_prefix(text: &str) -> &str {
    match text.find('\u{fffd}') {
        Some(idx) => &text[..idx],
        None => text,
    }
}

fn rebuild_state_for_current_turn(
    args: &Args,
    tokenizer: &Tokenizer,
    model: &mut LoadedModel,
    model_source: &ModelSource,
    state: &mut ChatState,
    current_input_ids: &[u32],
    device: &Device,
) -> Result<()> {
    let mut kept_turns = state.turns.clone();
    let mut rebuilt_tokens = encode_prompt(
        tokenizer,
        &build_prompt_from_history(&args.system, &kept_turns, None),
    )?;

    while !kept_turns.is_empty()
        && rebuilt_tokens.len() + current_input_ids.len() + args.sample_len > args.max_context
    {
        kept_turns.remove(0);
        rebuilt_tokens = encode_prompt(
            tokenizer,
            &build_prompt_from_history(&args.system, &kept_turns, None),
        )?;
    }

    if rebuilt_tokens.len() + current_input_ids.len() + args.sample_len > args.max_context {
        rebuilt_tokens.clear();
    }

    *model = model_source.load(device)?;
    state.turns = kept_turns;
    state.tokens.clear();

    if !rebuilt_tokens.is_empty() {
        let _ = model.forward_prompt(rebuilt_tokens.as_slice(), 0, device)?;
        state.tokens = rebuilt_tokens;
    }

    Ok(())
}

fn build_prompt_from_history(
    system: &str,
    turns: &[(String, String)],
    current_user: Option<&str>,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(system);
    prompt.push_str("<|im_end|>\n");

    for (user, assistant) in turns {
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(user);
        prompt.push_str("<|im_end|>\n");
        prompt.push_str("<|im_start|>assistant\n");
        prompt.push_str(assistant);
        prompt.push_str("<|im_end|>\n");
    }

    if let Some(user) = current_user {
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(user);
        prompt.push_str("<|im_end|>\n");
        prompt.push_str("<|im_start|>assistant\n");
    }

    prompt
}

fn load_config(path: &Path) -> Result<Config> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    serde_json::from_str(&data).context("failed to parse qwen2 config")
}

fn load_model(
    config: &Config,
    weights: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<ModelForCausalLM> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(weights, dtype, device) }
        .context("failed to memory-map safetensors weights")?;
    ModelForCausalLM::new(config, vb).context("failed to build qwen3 model")
}

fn load_quantized_model(path: &Path, device: &Device) -> Result<QuantizedModelWeights> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open gguf file {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .with_context(|| format!("failed to read gguf metadata from {}", path.display()))?;
    QuantizedModelWeights::from_gguf(content, &mut file, device)
        .context("failed to build quantized qwen3 model")
}

fn download_weight_files(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    let repo_info = repo.info().context("failed to fetch repository info")?;
    let filenames: BTreeSet<String> = repo_info
        .siblings
        .into_iter()
        .map(|entry| entry.rfilename)
        .collect();

    if filenames.contains("model.safetensors.index.json") {
        let index_path = repo
            .get("model.safetensors.index.json")
            .context("failed to fetch safetensors index")?;
        let index: WeightsIndex = serde_json::from_str(
            &fs::read_to_string(&index_path)
                .with_context(|| format!("failed to read {}", index_path.display()))?,
        )
        .context("failed to parse safetensors index")?;

        let shards: BTreeSet<String> = index.weight_map.into_values().collect();
        let mut paths = Vec::with_capacity(shards.len());
        for shard in shards {
            paths.push(
                repo.get(&shard)
                    .with_context(|| format!("failed to fetch shard {shard}"))?,
            );
        }
        return Ok(paths);
    }

    if filenames.contains("model.safetensors") {
        return Ok(vec![
            repo.get("model.safetensors")
                .context("failed to fetch model.safetensors")?,
        ]);
    }

    bail!("no safetensors weights found in repository")
}
