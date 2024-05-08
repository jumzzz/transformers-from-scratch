use std::os::unix::net::UnixDatagram;

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

/*

Note on Lifetime Elison:
The function signature now tells Rust that for some lifetime 'a, 
the function takes two parameters, both of which are string slices 
that live at least as long as lifetime 'a. The function signature also
tells Rust that the string slice returned from the function will live at 
least as long as lifetime 'a. In practice, it means that the lifetime of 
the reference returned by the longest function is the same as the smaller 
of the lifetimes of the values referred to by the function arguments. 
These relationships are what we want Rust to use when analyzing this code.

*/
struct TransformerWeights<'data> {
    token_embedding_table   : &'data [f32],
    rms_att_weight          : &'data [f32],
    rms_ffn_weight          : &'data [f32],
    wq                      : &'data [f32],
    wk                      : &'data [f32],
    wv                      : &'data [f32],
    wo                      : &'data [f32],
    w1                      : &'data [f32],
    w2                      : &'data [f32],
    w3                      : &'data [f32],
    rms_final_weight        : &'data [f32],
    wcls                    : &'data [f32],
}
struct RunState<'data> {
    x           : &'data [f32],
    xb          : &'data [f32], 
    xb2         : &'data [f32],
    hb          : &'data [f32],
    hb2         : &'data [f32],
    q           : &'data [f32],
    k           : &'data [f32],
    v           : &'data [f32],
    att         : &'data [f32],
    logits      : &'data [f32],
    // kv cache
    key_cache   : &'data [f32], 
    value_cache : &'data [f32],
}

struct Transformer<'data> {
    config  : Config,
    weights : TransformerWeights<'data>,
    state   : RunState<'data>,
    fd      : i32,          // File Descriptor for memory mapping?
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
