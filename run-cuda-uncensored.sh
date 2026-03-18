#model=${1:-unsloth/Qwen3.5-0.8B-GGUF}
model=${1:-DavidAU/Qwen3-4B-2507-Thinking-heretic-abliterated-uncensored}
cargo run --features cuda -- --sample-len 2048 --max-context 40960 --model $model --chat
