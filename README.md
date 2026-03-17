# qwen3-4b-runner

A small Rust CLI chat runner built on top of Candle for local Qwen models.

Current defaults target:

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-4B-GGUF`
- `Qwen3-4B-Q4_K_M.gguf`

The app supports:

- dense `safetensors` loading
- quantized `GGUF` loading
- interactive chat mode
- streaming output
- `vim` editing mode in chat input
- input history with up/down arrow
- `--max-context` trimming and cache rebuild

## Requirements

- Rust toolchain
- Windows with PowerShell and Git Bash
- For CUDA:
  - NVIDIA driver new enough for your installed CUDA toolkit
  - CUDA Toolkit installed
  - `source ./setenv.cuda` before running the CUDA build

## Quick Start

CPU quantized chat:

```bash
./run-cpu.sh
```

CUDA quantized chat:

```bash
source ./setenv.cuda
./run-cuda.sh
```

Manual run:

```bash
cargo run --features cuda -- --quantized --chat
```

## Common Commands

Show help:

```bash
cargo run -- --help
```

Single prompt on CPU:

```bash
cargo run -- --cpu --quantized --prompt "Explain Rust ownership in Traditional Chinese."
```

Chat with a larger generation budget:

```bash
cargo run --features cuda -- --quantized --chat --sample-len 512 --max-context 4096
```

## CLI Options

Important options:

- `--chat`: start interactive chat mode
- `--prompt`: run a single-turn prompt
- `--quantized`: load the GGUF model instead of dense safetensors
- `--sample-len`: max newly generated tokens
- `--max-context`: retained context budget, default `4096`
- `--cpu`: force CPU even if CUDA feature is enabled

Default model options:

- `--model Qwen/Qwen3-4B`
- `--quantized-model Qwen/Qwen3-4B-GGUF`
- `--gguf-file Qwen3-4B-Q4_K_M.gguf`

## Chat UX

Interactive chat uses `rustyline` with `vim` mode enabled.

Useful keys:

- `Esc`: switch to normal mode
- `i`, `a`: return to insert mode
- `h`, `j`, `k`, `l`: move in normal mode
- `Up` / `Down`: recall previous inputs
- `Ctrl+D`: exit chat

## CUDA Notes

`setenv.cuda` is a Bash helper that exports CUDA, MSVC, and Windows SDK environment variables.

Typical flow:

```bash
source ./setenv.cuda
nvidia-smi
cargo run --features cuda -- --quantized --chat
```

If your driver is older than the installed CUDA toolkit supports, you can get runtime errors such as unsupported PTX/toolchain issues. In that case, upgrade the driver or install an older toolkit.

## Model Cache

Downloaded Hugging Face files are stored in the local HF cache, typically:

```text
C:\Users\<user>\.cache\huggingface\hub
```

## Status

This project currently targets `Qwen3-4B` cleanly on Candle.

`Qwen3.5-4B` is not wired in because it uses a different architecture than the current `qwen3` Candle path.
