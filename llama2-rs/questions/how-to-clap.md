#### How to Clap Example

Here's the working implementation of Command-line argument using `clap`

```rust
use clap::{Parser,ValueEnum};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    Generate,
    Chat,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Config {
    #[arg(long, required = true)]
    checkpoint_path: String,
    #[arg(long, default_value = "tokenizer.bin")]
    tokenizer_path: String,
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    topp: f32,
    #[arg(long, default_value_t = 256)]
    steps: u32,
    #[arg(long, default_value = "")]
    prompt: String,
    #[arg(long, default_value_t = 0)]
    rng_seed: u64,
    #[arg(value_enum, default_value_t = Mode::Generate)]
    mode: Mode,
    #[arg(long, default_value = "")]
    system_prompt: String,
}

fn main() {
    let config = Config::parse();

    println!("Checkpoint Path: {}", config.checkpoint_path);
    println!("Tokenizer Path: {}", config.tokenizer_path);
    println!("Temperature: {}", config.temperature);
    println!("Top-p: {}", config.topp);
    println!("Steps: {}", config.steps);
    println!("Prompt: '{}'", config.prompt);
    println!("RNG Seed: {}", config.rng_seed);
    println!("Mode: {:?}", config.mode);
    println!("System Prompt: '{}'", config.system_prompt);
}

```