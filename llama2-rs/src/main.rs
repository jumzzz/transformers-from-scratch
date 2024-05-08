use clap::{Parser,ValueEnum};

// I don't think Vec<f32> is an optimal choice here
// But let's worry about that later
type FloatTensor = Vec<f32>;


struct Config {
    dim : i32,
    hidden_dim : i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

struct TransformerWeights {
    token_embedding_table   : FloatTensor,
    rms_att_weight          : FloatTensor,
    rms_ffn_weight          : FloatTensor,
    wq                      : FloatTensor,
    wk                      : FloatTensor,
    wv                      : FloatTensor,
    wo                      : FloatTensor,
    w1                      : FloatTensor,
    w2                      : FloatTensor,
    w3                      : FloatTensor,
    rms_final_weight        : FloatTensor,
    wcls                    : FloatTensor,
}

struct RunState {
    x           : FloatTensor,
    xb          : FloatTensor, 
    xb2         : FloatTensor,
    hb          : FloatTensor,
    hb2         : FloatTensor,
    q           : FloatTensor,
    k           : FloatTensor,
    v           : FloatTensor,
    att         : FloatTensor,
    logits      : FloatTensor,
    // kv cache
    key_cache   : FloatTensor, 
    value_cache : FloatTensor,
}

struct Transformer {
    config  : Config,
    weights : TransformerWeights,
    state   : RunState,
}


#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    Generate,
    Chat,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
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
    let config = Cli::parse();

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
