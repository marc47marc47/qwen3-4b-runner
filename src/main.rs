use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use candle_transformers::utils::apply_repeat_penalty;
use clap::Parser;
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen2.5-7B-Instruct";
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model: String,

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

    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,

    #[arg(long)]
    cpu: bool,
}

#[derive(Debug, Deserialize)]
struct WeightsIndex {
    weight_map: std::collections::HashMap<String, String>,
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
    eprintln!("model: {}", args.model);

    let repo = Api::new()
        .context("failed to create Hugging Face API client")?
        .model(args.model.clone());

    let config_path = repo.get("config.json").context("failed to fetch config.json")?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("failed to fetch tokenizer.json")?;
    let weight_paths = download_weight_files(&repo)
        .with_context(|| format!("failed to fetch weights for {}", args.model))?;

    let config = load_config(&config_path)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let mut model = load_model(&config, &weight_paths, dtype, &device)?;

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
    model: &mut ModelForCausalLM,
    logits_processor: &mut LogitsProcessor,
    eos_token: Option<u32>,
    device: &Device,
) -> Result<()> {
    let mut state = ChatState::new();
    let mut line = String::new();
    let mut first_turn = true;

    println!("chat mode: type 'exit' or 'quit' to leave");

    loop {
        print!("you> ");
        io::stdout().flush().context("failed to flush stdout")?;
        line.clear();
        let bytes = io::stdin()
            .read_line(&mut line)
            .context("failed to read from stdin")?;
        if bytes == 0 {
            println!();
            break;
        }

        let user_input = line.trim();
        if user_input.is_empty() {
            continue;
        }
        if matches!(user_input, "exit" | "quit") {
            break;
        }

        let prompt = if first_turn {
            build_initial_prompt(&args.system, user_input)
        } else {
            build_turn_prompt(user_input)
        };
        first_turn = false;

        let input_ids = encode_prompt(tokenizer, &prompt)?;
        print!("assistant> ");
        io::stdout().flush().context("failed to flush stdout")?;
        let generated = generate_reply(
            args,
            tokenizer,
            model,
            logits_processor,
            &mut state,
            &input_ids,
            eos_token,
            device,
            true,
        )?;
        if generated.is_empty() {
            println!();
        }
    }

    Ok(())
}

fn generate_reply(
    args: &Args,
    tokenizer: &Tokenizer,
    model: &mut ModelForCausalLM,
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

    eprintln!("prompt_tokens: {}", input_ids.len());
    eprintln!("sampling {} new tokens...", args.sample_len);

    let prefill = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let logits = model.forward(&prefill, state.tokens.len())?;
    let mut logits = logits.squeeze(0)?.squeeze(0)?;
    state.tokens.extend_from_slice(input_ids);

    let generated_start = state.tokens.len();
    let mut visible_text = String::new();

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
            let piece = tokenizer
                .decode(&[next_token], false)
                .map_err(anyhow::Error::msg)?;
            print!("{piece}");
            io::stdout().flush().context("failed to flush stdout")?;
            visible_text.push_str(&piece);
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
        println!();
        visible_text
    } else {
        tokenizer
            .decode(generated_tokens, true)
            .map_err(anyhow::Error::msg)?
    };
    Ok(generated)
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
    ModelForCausalLM::new(config, vb).context("failed to build qwen2 model")
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
