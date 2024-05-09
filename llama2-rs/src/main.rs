use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io;

use std::mem;
use clap::{Parser,ValueEnum};
use std::io::ErrorKind;

// I don't think Vec<f32> is an optimal choice here
// But let's worry about that later


struct Config {
    dim : i32,
    hidden_dim : i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

impl Config {
    fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 28 {
            return Err(io::Error::new(ErrorKind::Other, "Insufficient bytes for Config"));
        }
        // Define the closure for converting bytes to i32
        let to_i32 = |b: &[u8]| -> io::Result<i32> {
            b.try_into()
                .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid byte slice"))
                .map(i32::from_ne_bytes)
        };
        
        Ok(Config {
            dim: to_i32(&bytes[0..4])?,
            hidden_dim: to_i32(&bytes[4..8])?,
            n_layers: to_i32(&bytes[8..12])?,
            n_heads: to_i32(&bytes[12..16])?,
            n_kv_heads: to_i32(&bytes[16..20])?,
            vocab_size: to_i32(&bytes[20..24])?,
            seq_len: to_i32(&bytes[24..28])?,
        })
    }
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

fn main() -> io::Result<()>  {
    let cli = Cli::parse();

    println!("Checkpoint Path: {}", cli.checkpoint_path);
    println!("Tokenizer Path: {}", cli.tokenizer_path);
    println!("Temperature: {}", cli.temperature);
    println!("Top-p: {}", cli.topp);
    println!("Steps: {}", cli.steps);
    println!("Prompt: '{}'", cli.prompt);
    println!("RNG Seed: {}", cli.rng_seed);
    println!("Mode: {:?}", cli.mode);
    println!("System Prompt: '{}'", cli.system_prompt);
    
    let file = File::open(cli.checkpoint_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let config_size = mem::size_of::<Config>();
    let config = Config::from_bytes(&mmap)?;

    Ok(())
}
