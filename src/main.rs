use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedModelWeights;
use candle_transformers::models::qwen3::{
    Config as Qwen3Config, ModelForCausalLM as Qwen3ModelForCausalLM,
};
use candle_transformers::utils::apply_repeat_penalty;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use hf_hub::api::sync::Api;
use rustyline::DefaultEditor;
use rustyline::config::{CompletionType, Config as RustylineConfig, EditMode};
use rustyline::error::ReadlineError;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

mod quantized_qwen3_5;
mod qwen3_5;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-4B";
const DEFAULT_GGUF_FILE: &str = "Qwen3-4B-Q4_K_M.gguf";
const DEFAULT_SYSTEM_PROMPT: &str = "避免大陸用語，避免提到'中國'. 喜歡美食，旅遊，地理，人文，程式以及資料分析，回答簡潔扼要，追求事實";

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model: String,

    #[arg(long, default_value = DEFAULT_GGUF_FILE)]
    gguf_file: String,

    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system: String,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    chat: bool,

    #[arg(long)]
    no_system: bool,

    #[arg(long)]
    raw_prompt: bool,

    #[arg(long, default_value_t = 4096)]
    sample_len: usize,

    #[arg(long, default_value_t = 0.3)]
    temperature: f64,

    #[arg(long, default_value_t = 0.85)]
    top_p: f64,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value_t = 40960)]
    max_context: usize,

    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    debug_tokens: bool,
}

#[derive(Debug, Deserialize)]
struct WeightsIndex {
    weight_map: HashMap<String, String>,
}

enum LoadedModel {
    DenseQwen3(Qwen3ModelForCausalLM),
    DenseQwen3_5(qwen3_5::ModelForCausalLM),
    Quantized(QuantizedModelWeights),
    QuantizedQwen3_5(quantized_qwen3_5::ModelWeights),
}

impl LoadedModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::DenseQwen3(model) => model.forward(input, index_pos),
            Self::DenseQwen3_5(model) => model.forward(input, index_pos),
            Self::Quantized(model) => model.forward(input, index_pos),
            Self::QuantizedQwen3_5(model) => model.forward(input, index_pos),
        }
    }

    fn forward_prompt(
        &mut self,
        input_ids: &[u32],
        index_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        match self {
            Self::DenseQwen3(model) => {
                let input = Tensor::new(input_ids, device)?.unsqueeze(0)?;
                Ok(model.forward(&input, index_pos)?)
            }
            Self::DenseQwen3_5(model) => {
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
            Self::QuantizedQwen3_5(model) => {
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
                logits.context("quantized qwen3.5 forward received empty prompt")
            }
        }
    }
}

#[derive(Clone)]
enum DenseModelConfig {
    Qwen3(Qwen3Config),
    Qwen3_5(qwen3_5::Config),
}

#[derive(Clone)]
enum ModelSource {
    Dense {
        repo_id: String,
        config: DenseModelConfig,
        weights: Vec<PathBuf>,
        dtype: DType,
    },
    Quantized {
        repo_id: String,
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
                ..
            } => load_model(config, weights, *dtype, device),
            Self::Quantized { gguf_path, .. } => load_quantized_model(gguf_path, device),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PromptKind<'a> {
    Initial { system: &'a str, user: &'a str },
    NextTurn { user: &'a str },
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let device = select_device(args.cpu)?;
    let dtype = preferred_dtype(&device);
    let api = Api::new().context("failed to create Hugging Face API client")?;
    let model_source = resolve_model_source(&api, &args, dtype)?;
    let (tokenizer, tokenizer_repo_id) = load_tokenizer(&api, &args.model, &model_source)?;
    log_runtime_configuration(&device, dtype, &model_source, &tokenizer_repo_id);

    let mut model = model_source.load(&device)?;

    let eos_token = tokenizer.token_to_id("<|im_end|>");
    if args.debug_tokens {
        eprintln!("debug: eos_token(<|im_end|>) = {eos_token:?}");
    }
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
        let prompt = build_prompt_for_args(&args, PromptKind::Initial {
            system: &args.system,
            user: args.prompt.as_deref().expect("validated above"),
        });
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

fn validate_args(args: &Args) -> Result<()> {
    if !args.chat && args.prompt.is_none() {
        bail!("--prompt is required unless --chat is enabled");
    }

    if args.raw_prompt && args.chat {
        bail!("--raw-prompt cannot be used with --chat");
    }

    Ok(())
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

fn log_runtime_configuration(
    device: &Device,
    dtype: DType,
    model_source: &ModelSource,
    tokenizer_repo_id: &str,
) {
    eprintln!("device: {device:?}");
    eprintln!("dtype: {dtype:?}");
    match model_source {
        ModelSource::Dense { repo_id, .. } => eprintln!("model: {repo_id}"),
        ModelSource::Quantized { repo_id, .. } => eprintln!("model: {repo_id}"),
    }
    eprintln!("tokenizer: {tokenizer_repo_id}");
}

fn load_tokenizer(
    api: &Api,
    requested_model_id: &str,
    model_source: &ModelSource,
) -> Result<(Tokenizer, String)> {
    let candidates = tokenizer_model_candidates(requested_model_id, model_source);
    for model_id in &candidates {
        let repo = api.model(model_id.clone());
        if let Ok(tokenizer_path) = repo.get("tokenizer.json") {
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
            return Ok((tokenizer, model_id.clone()));
        }
    }

    bail!(
        "failed to fetch tokenizer.json from any candidate repo: {}",
        candidates.join(", ")
    )
}

fn tokenizer_model_candidates(requested_model_id: &str, model_source: &ModelSource) -> Vec<String> {
    let mut candidates = Vec::new();
    push_unique(&mut candidates, requested_model_id.to_string());
    if let Some(official) = infer_official_qwen_repo(requested_model_id) {
        push_unique(&mut candidates, official);
    }

    match model_source {
        ModelSource::Dense { repo_id, .. } => {
            push_unique(&mut candidates, repo_id.clone());
            if let Some(official) = infer_official_qwen_repo(repo_id) {
                push_unique(&mut candidates, official);
            }
        }
        ModelSource::Quantized { repo_id, .. } => {
            push_unique(&mut candidates, repo_id.clone());
            if let Some(official) = infer_official_qwen_repo(repo_id) {
                push_unique(&mut candidates, official);
            }
            if let Some(stripped) = strip_packaging_suffixes(requested_model_id) {
                if let Some(official) = infer_official_qwen_repo(&stripped) {
                    push_unique(&mut candidates, official);
                }
                push_unique(&mut candidates, stripped);
            }
            if let Some(stripped) = strip_packaging_suffixes(repo_id) {
                if let Some(official) = infer_official_qwen_repo(&stripped) {
                    push_unique(&mut candidates, official);
                }
                push_unique(&mut candidates, stripped);
            }
        }
    }

    candidates
}

fn strip_packaging_suffixes(model_id: &str) -> Option<String> {
    const SUFFIXES: [&str; 4] = ["-GGUF", "-gguf", "-Imatrix", "-imatrix"];

    let mut stripped = model_id.to_string();
    let mut changed = false;

    loop {
        let mut removed = false;
        for suffix in SUFFIXES {
            if let Some(next) = stripped.strip_suffix(suffix) {
                stripped = next.to_string();
                changed = true;
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }

    changed.then_some(stripped)
}

fn infer_official_qwen_repo(model_id: &str) -> Option<String> {
    const PREFIXES: [&str; 2] = ["Qwen3.5-", "Qwen3-"];

    for prefix in PREFIXES {
        if let Some(start) = model_id.find(prefix) {
            let tail = &model_id[start..];
            let size = tail[prefix.len()..]
                .chars()
                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '.')
                .collect::<String>();
            if size.is_empty() {
                continue;
            }
            let model_name = format!("{prefix}{size}");
            return Some(format!("Qwen/{model_name}"));
        }
    }

    None
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn resolve_model_source(api: &Api, args: &Args, dtype: DType) -> Result<ModelSource> {
    let repo_id = args.model.clone();
    let repo = api.model(repo_id.clone());
    let repo_info = repo.info().context("failed to fetch repository info")?;
    let filenames: BTreeSet<String> = repo_info
        .siblings
        .into_iter()
        .map(|entry| entry.rfilename)
        .collect();

    let has_dense = filenames.contains("config.json")
        && (filenames.contains("model.safetensors")
            || filenames.contains("model.safetensors.index.json"));
    let gguf_files: Vec<&String> = filenames
        .iter()
        .filter(|name| name.ends_with(".gguf"))
        .collect();
    let has_gguf = !gguf_files.is_empty();

    if has_dense && has_gguf {
        if filenames.contains(&args.gguf_file) {
            let gguf_path = repo
                .get(&args.gguf_file)
                .with_context(|| format!("failed to fetch gguf file {}", args.gguf_file))?;
            return Ok(ModelSource::Quantized { repo_id, gguf_path });
        }
    }

    if has_dense {
        let config_path = repo
            .get("config.json")
            .context("failed to fetch config.json")?;
        let weight_paths = download_weight_files(&repo)
            .with_context(|| format!("failed to fetch weights for {}", repo_id))?;
        let config = load_config(&config_path)?;
        return Ok(ModelSource::Dense {
            repo_id,
            config,
            weights: weight_paths,
            dtype,
        });
    }

    if has_gguf {
        let gguf_name = if filenames.contains(&args.gguf_file) {
            args.gguf_file.clone()
        } else if gguf_files.len() == 1 {
            gguf_files[0].clone().to_string()
        } else {
            choose_gguf_file(&repo_id, &gguf_files)?
                .ok_or_else(|| anyhow::anyhow!("gguf selection cancelled"))?
        };
        let gguf_path = repo
            .get(&gguf_name)
            .with_context(|| format!("failed to fetch gguf file {}", gguf_name))?;
        return Ok(ModelSource::Quantized { repo_id, gguf_path });
    }

    bail!(
        "repo {} does not contain a supported dense or gguf layout",
        repo_id
    )
}

fn choose_gguf_file(repo_id: &str, gguf_files: &[&String]) -> Result<Option<String>> {
    let option_names = gguf_files
        .iter()
        .map(|name| name.as_str())
        .filter(|name| !name.starts_with("mmproj-"))
        .collect::<Vec<_>>();
    let option_names = if option_names.is_empty() {
        gguf_files
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>()
    } else {
        option_names
    };
    let labels = selection_labels(option_names.len())?;
    let preferred_index = preferred_gguf_index(&option_names);
    let sizes = fetch_gguf_sizes(repo_id, &option_names);

    println!("repo {repo_id} contains multiple gguf files");
    println!("Press one key to choose, or Esc to cancel.");
    for (index, option) in option_names.iter().enumerate() {
        let size = sizes
            .get(*option)
            .and_then(|size| *size)
            .map(format_bytes)
            .unwrap_or_else(|| "-".to_string());
        let preferred = if index == preferred_index {
            " (default)"
        } else {
            ""
        };
        println!("  [{}] {:>8}  {}{}", labels[index], size, option, preferred);
    }
    enable_raw_mode().context("failed to enable terminal raw mode")?;

    loop {
        match event::read().context("failed to read terminal event")? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Char(ch) => {
                    let input = ch.to_ascii_lowercase().to_string();
                    if let Some(index) = labels.iter().position(|label| label == &input) {
                        disable_raw_mode().context("failed to disable terminal raw mode")?;
                        println!("selected: {}", option_names[index]);
                        return Ok(Some(option_names[index].to_string()));
                    }
                }
                KeyCode::Esc => {
                    disable_raw_mode().context("failed to disable terminal raw mode")?;
                    println!("selection cancelled");
                    return Ok(None);
                }
                _ => {}
            },
            _ => {}
        }
    }
}

fn selection_labels(count: usize) -> Result<Vec<String>> {
    const LABELS: &str = "1234567890abcdefghijklmnopqrstuvwxyz";
    let labels = LABELS
        .chars()
        .take(count)
        .map(|ch| ch.to_string())
        .collect::<Vec<_>>();
    if labels.len() != count {
        bail!("too many gguf files to display in a single-key menu ({count})");
    }
    Ok(labels)
}

fn preferred_gguf_index(options: &[&str]) -> usize {
    options
        .iter()
        .position(|name| name.ends_with("Q4_K_M.gguf"))
        .or_else(|| {
            options
                .iter()
                .position(|name| name.ends_with("Q4_K_S.gguf"))
        })
        .or_else(|| options.iter().position(|name| name.ends_with("Q4_0.gguf")))
        .unwrap_or(0)
}

fn fetch_gguf_sizes(repo_id: &str, option_names: &[&str]) -> HashMap<String, Option<u64>> {
    option_names
        .iter()
        .map(|name| ((*name).to_string(), fetch_file_size(repo_id, name)))
        .collect()
}

fn fetch_file_size(repo_id: &str, filename: &str) -> Option<u64> {
    let url = format!(
        "https://huggingface.co/{repo_id}/resolve/main/{}",
        filename.replace(' ', "%20")
    );
    let response = ureq::head(&url).call().ok()?;
    response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
}

fn format_bytes(size: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = size as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{size} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn build_prompt(kind: PromptKind<'_>) -> String {
    match kind {
        PromptKind::Initial { system, user } => format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ),
        PromptKind::NextTurn { user } => {
            format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        }
    }
}

fn build_prompt_for_args(args: &Args, kind: PromptKind<'_>) -> String {
    match kind {
        PromptKind::Initial { system, user } => {
            if args.raw_prompt {
                user.to_string()
            } else if args.no_system {
                format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
            } else {
                build_prompt(PromptKind::Initial { system, user })
            }
        }
        PromptKind::NextTurn { user } => build_prompt(PromptKind::NextTurn { user }),
    }
}

fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(anyhow::Error::msg)?;
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
            build_prompt_for_args(args, PromptKind::Initial {
                system: &args.system,
                user: user_input,
            })
        } else {
            build_prompt(PromptKind::NextTurn { user: user_input })
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
        state
            .turns
            .push((user_input.to_string(), generated.clone()));
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
        rebuild_state_for_current_turn(
            args,
            tokenizer,
            model,
            model_source,
            state,
            input_ids,
            device,
        )?;
    }

    if state.tokens.len() + input_ids.len() > args.max_context {
        bail!(
            "prompt exceeds --max-context ({} tokens). Reduce prompt size or raise the limit.",
            args.max_context
        );
    }

    eprintln!("prompt_tokens: {}", input_ids.len());
    eprintln!("sampling {} new tokens...", args.sample_len);

    let prefill_started = Instant::now();
    let logits = model.forward_prompt(input_ids, state.tokens.len(), device)?;
    let prefill_elapsed = prefill_started.elapsed();
    let mut logits = logits.squeeze(0)?.squeeze(0)?;
    state.tokens.extend_from_slice(input_ids);

    let generated_start = state.tokens.len();
    let mut visible_text = String::new();
    let mut streamed_tokens = Vec::new();
    let mut sampled_tokens = Vec::new();
    let decode_started = Instant::now();

    for _ in 0..args.sample_len {
        let logits_with_penalty = if args.repeat_penalty > 1.0 {
            let start_at = state.tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&logits, args.repeat_penalty, &state.tokens[start_at..])?
        } else {
            logits.clone()
        };

        let next_token = logits_processor.sample(&logits_with_penalty)?;
        state.tokens.push(next_token);
        sampled_tokens.push(next_token);
        if eos_token.is_some_and(|id| id == next_token) {
            if args.debug_tokens {
                eprintln!("debug: stopping on eos token {next_token}");
            }
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

    if args.debug_tokens {
        let sampled_text = tokenizer
            .decode(&sampled_tokens, true)
            .map_err(anyhow::Error::msg)
            .unwrap_or_else(|_| "<decode error>".to_string());
        let visible_text_debug = tokenizer
            .decode(generated_tokens, true)
            .map_err(anyhow::Error::msg)
            .unwrap_or_else(|_| "<decode error>".to_string());
        eprintln!("debug: sampled_token_ids = {:?}", sampled_tokens);
        eprintln!("debug: sampled_text = {:?}", sampled_text);
        eprintln!("debug: visible_token_ids = {:?}", generated_tokens);
        eprintln!("debug: visible_text = {:?}", visible_text_debug);
    }

    let decode_elapsed = decode_started.elapsed();
    eprintln!(
        "prefill: {:.2}s, {:.2} tok/s",
        prefill_elapsed.as_secs_f64(),
        tokens_per_second(input_ids.len(), prefill_elapsed)
    );
    eprintln!(
        "decode: {} tokens in {:.2}s, {:.2} tok/s",
        generated_tokens.len(),
        decode_elapsed.as_secs_f64(),
        tokens_per_second(generated_tokens.len(), decode_elapsed)
    );
    Ok(generated)
}

fn stable_prefix(text: &str) -> &str {
    match text.find('\u{fffd}') {
        Some(idx) => &text[..idx],
        None => text,
    }
}

fn tokens_per_second(tokens: usize, elapsed: std::time::Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= f64::EPSILON {
        0.0
    } else {
        tokens as f64 / secs
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
        &build_prompt_from_history(
            if args.no_system {
                None
            } else {
                Some(args.system.as_str())
            },
            &kept_turns,
            None,
        ),
    )?;

    while !kept_turns.is_empty()
        && rebuilt_tokens.len() + current_input_ids.len() + args.sample_len > args.max_context
    {
        kept_turns.remove(0);
        rebuilt_tokens = encode_prompt(
            tokenizer,
            &build_prompt_from_history(
                if args.no_system {
                    None
                } else {
                    Some(args.system.as_str())
                },
                &kept_turns,
                None,
            ),
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
    system: Option<&str>,
    turns: &[(String, String)],
    current_user: Option<&str>,
) -> String {
    let mut prompt = String::new();
    if let Some(system) = system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
    }

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

fn load_config(path: &Path) -> Result<DenseModelConfig> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    let config_json: serde_json::Value =
        serde_json::from_str(&data).context("failed to parse model config json")?;

    if config_json.get("text_config").is_some() {
        let config: qwen3_5::Config =
            serde_json::from_value(config_json).context("failed to parse qwen3.5 config")?;
        return Ok(DenseModelConfig::Qwen3_5(config));
    }

    let config: Qwen3Config =
        serde_json::from_str(&data).context("failed to parse qwen3 config")?;
    Ok(DenseModelConfig::Qwen3(config))
}

fn load_model(
    config: &DenseModelConfig,
    weights: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<LoadedModel> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(weights, dtype, device) }
        .context("failed to memory-map safetensors weights")?;

    match config {
        DenseModelConfig::Qwen3(config) => Qwen3ModelForCausalLM::new(config, vb)
            .map(LoadedModel::DenseQwen3)
            .context("failed to build qwen3 model"),
        DenseModelConfig::Qwen3_5(config) => qwen3_5::ModelForCausalLM::new(config, vb)
            .map(LoadedModel::DenseQwen3_5)
            .context("failed to build qwen3.5 model"),
    }
}

fn load_quantized_model(path: &Path, device: &Device) -> Result<LoadedModel> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open gguf file {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file).map_err(|err| {
        annotate_gguf_read_error(err, path).context(format!(
            "failed to read gguf metadata from {}",
            path.display()
        ))
    })?;
    let summary = quantized_qwen3_5::MetadataSummary::detect(&content.metadata);
    match summary.architecture {
        quantized_qwen3_5::QuantizedArchitecture::Qwen3 => {
            QuantizedModelWeights::from_gguf(content, &mut file, device)
                .map(LoadedModel::Quantized)
                .context("failed to build quantized qwen3 model")
        }
        quantized_qwen3_5::QuantizedArchitecture::Qwen3_5 => {
            quantized_qwen3_5::ModelWeights::from_gguf(content, &mut file, device)
                .map(LoadedModel::QuantizedQwen3_5)
                .context("failed to build quantized qwen3.5 model")
        }
        quantized_qwen3_5::QuantizedArchitecture::Unknown => bail!(
            "unsupported gguf architecture. general.architecture={:?}, metadata_prefix={:?}",
            summary.general_architecture,
            summary.metadata_prefix
        ),
    }
}

fn annotate_gguf_read_error(err: candle_core::Error, path: &Path) -> anyhow::Error {
    let message = err.to_string();
    if let Some(dtype) = parse_unknown_gguf_dtype(&message) {
        let dtype_name = ggml_dtype_name(dtype).unwrap_or("unknown/newer ggml dtype");
        let supported = "supported candle-core 0.9.2 dtypes: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K";
        return anyhow::anyhow!(
            "gguf file {} uses unsupported ggml dtype {} ({dtype_name}). {supported}",
            path.display(),
            dtype
        )
        .context("this is a candle-core GGUF parser limitation, not a qwen3.5 loader bug");
    }
    err.into()
}

fn parse_unknown_gguf_dtype(message: &str) -> Option<u32> {
    let marker = "unknown dtype for tensor ";
    let idx = message.find(marker)?;
    message[idx + marker.len()..]
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn ggml_dtype_name(dtype: u32) -> Option<&'static str> {
    match dtype {
        16 => Some("IQ2_XXS"),
        17 => Some("IQ2_XS"),
        18 => Some("IQ3_XXS"),
        19 => Some("IQ1_S"),
        20 => Some("IQ4_NL"),
        21 => Some("IQ3_S"),
        22 => Some("IQ2_S"),
        23 => Some("IQ4_XS"),
        24 => Some("I8"),
        25 => Some("I16"),
        26 => Some("I32"),
        27 => Some("I64"),
        28 => Some("F64"),
        29 => Some("IQ1_M"),
        30 => Some("BF16"),
        _ => None,
    }
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
