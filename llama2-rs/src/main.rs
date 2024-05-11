use memmap2::{Mmap, MmapOptions};
use core::slice;
use std::fs::File;
use std::io;

use std::mem;
use clap::{Parser,ValueEnum};
use std::io::ErrorKind;

// 
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

impl Config {
    fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 28 {
            return Err(io::Error::new(ErrorKind::Other, "Insufficient bytes for Config"));
        }
        // Define the closure for converting bytes to i32
        let to_i32 = |b: &[u8]| -> io::Result<i32> {
            b.try_into().map_err(
                |_| io::Error::new(ErrorKind::InvalidData, "Invalid byte slice")
            ).map(i32::from_le_bytes)
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
struct TransformerWeights <'a>{
    token_embedding_table   : &'a [f32], 
    rms_att_weight          : &'a [f32],
    rms_ffn_weight          : &'a [f32],
    wq                      : &'a [f32],
    wk                      : &'a [f32],
    wv                      : &'a [f32],
    wo                      : &'a [f32],
    w1                      : &'a [f32],
    w2                      : &'a [f32],
    w3                      : &'a [f32],
    rms_final_weight        : &'a [f32],
    wcls                    : &'a [f32],
}

impl<'a> TransformerWeights <'a> {
    fn new(data: &'a Mmap, config: &Config) -> Self {
        unsafe {
            let dim = config.dim as usize;
            let hidden_dim = config.hidden_dim as usize;
            let n_layers = config.n_layers as usize;
            let n_heads = config.n_heads as usize;
            let n_kv_heads = config.n_kv_heads as usize;
            let vocab_size = config.vocab_size as usize;
            let seq_len = config.seq_len as usize;

            let shared_weights = if vocab_size > 0 { true } else { false };

            let head_size = dim / n_heads;
            const FLOAT_SIZE: usize = 4; 

            let start = mem::size_of::<Config>();
            let offset = start + vocab_size * dim * FLOAT_SIZE;
            let token_embedding_table = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * FLOAT_SIZE;
            let rms_att_weight = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * n_heads * head_size * FLOAT_SIZE;
            let wq = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * n_kv_heads * head_size * FLOAT_SIZE;
            let wk = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * n_kv_heads * head_size * FLOAT_SIZE;
            let wv = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * n_heads * head_size * dim * FLOAT_SIZE;
            let wo = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * FLOAT_SIZE;
            let rms_ffn_weight = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * hidden_dim * FLOAT_SIZE;
            let w1 = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * hidden_dim * dim * FLOAT_SIZE;
            let w2 = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * hidden_dim * FLOAT_SIZE;
            let w3 = &data[start..offset];

            let start = offset;
            let offset = start + dim * FLOAT_SIZE;  // Skipping rms_final_weight for now
            let rms_final_weight = &data[start..offset];

            let start = offset;
            let offset = start + seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_real
            let start = offset;
            let offset = start + seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_imag

            let wcls = if shared_weights {
                slice::from_raw_parts(token_embedding_table.as_ptr() as *const f32, token_embedding_table.len() / FLOAT_SIZE)
            } else {
                let wcls_raw = &data[offset..];
                slice::from_raw_parts(wcls_raw.as_ptr() as *const f32, wcls_raw.len() / FLOAT_SIZE)
            };

            TransformerWeights {
                token_embedding_table: slice::from_raw_parts(token_embedding_table.as_ptr() as *const f32, token_embedding_table.len() / FLOAT_SIZE),
                rms_att_weight: slice::from_raw_parts(rms_att_weight.as_ptr() as *const f32, rms_att_weight.len() / FLOAT_SIZE),
                rms_ffn_weight : slice::from_raw_parts(rms_ffn_weight.as_ptr() as *const f32, rms_ffn_weight.len() / FLOAT_SIZE),
                wq: slice::from_raw_parts(wq.as_ptr() as *const f32, wq.len() / FLOAT_SIZE),
                wk: slice::from_raw_parts(wk.as_ptr() as *const f32, wk.len() / FLOAT_SIZE),
                wv: slice::from_raw_parts(wv.as_ptr() as *const f32, wv.len() / FLOAT_SIZE),
                wo: slice::from_raw_parts(wo.as_ptr() as *const f32, wv.len() / FLOAT_SIZE),
                w1: slice::from_raw_parts(w1.as_ptr() as *const f32, w1.len() / FLOAT_SIZE),
                w2: slice::from_raw_parts(w2.as_ptr() as *const f32, w2.len() / FLOAT_SIZE),
                w3: slice::from_raw_parts(w3.as_ptr() as *const f32, w3.len() / FLOAT_SIZE),
                wcls,       
            } 
        }
    }
}

struct Transformer {
    config  : Config,
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
    let transformer_weights = TransformerWeights::new(&mmap, &config);



    Ok(())
}
