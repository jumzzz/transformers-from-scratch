use memmap2::{Mmap, MmapOptions};
use std::slice;
use std::fs::File;
use std::io;

use std::mem;
use clap::{Parser,ValueEnum};
use std::io::ErrorKind;

const FLOAT_SIZE: usize = 4;
const CONFIG_SIZE_IN_BYTES: usize = 28;

struct Config {
    dim : usize,
    hidden_dim : usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
}

impl Config {
    fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 28 {
            return Err(io::Error::new(ErrorKind::Other, "Insufficient bytes for Config"));
        }
        // Model files defined the bytes in terms of i32. Hence, before I convert them
        // to its respective usize equivalent, the bytes should be parsed first as i32 
        let to_usize = |b: &[u8]| -> io::Result<usize> {
            let num = b.try_into().map_err(
                |_| io::Error::new(ErrorKind::InvalidData, "Invalid byte slice")
            ).map(i32::from_le_bytes)?;
            Ok(num as usize)
        };

        Ok(Config {
            dim: to_usize(&bytes[0..4])?,
            hidden_dim: to_usize(&bytes[4..8])?,
            n_layers: to_usize(&bytes[8..12])?,
            n_heads: to_usize(&bytes[12..16])?,
            n_kv_heads: to_usize(&bytes[16..20])?,
            vocab_size: to_usize(&bytes[20..24])?,
            seq_len: to_usize(&bytes[24..28])?,
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
    fn new(data: &'a Mmap, cfg: &Config) -> Self {
        unsafe {
            // let dim = cfg.dim;
            // let hidden_dim = cfg.hidden_dim;
            // let n_layers = cfg.n_layers;
            // let n_heads = cfg.n_heads;
            // let n_kv_heads = cfg.n_kv_heads;
            // let vocab_size = cfg.vocab_size;
            // let seq_len = cfg.seq_len;

            let shared_weights = if cfg.vocab_size > 0 { true } else { false };

            let head_size = dim / n_heads;

            let start = CONFIG_SIZE_IN_BYTES;
            let offset = start + vocab_size * dim * FLOAT_SIZE;
            println!("(token_embedding_table) start = {}, offset = {}", start, offset);
            let token_embedding_table = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * FLOAT_SIZE;
            println!("(rms_att_weight) start = {}, offset = {}", start, offset);            
            let rms_att_weight = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * n_heads * head_size * FLOAT_SIZE;
            println!("(wq) start = {}, offset = {}", start, offset);            
            let wq = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * n_kv_heads * head_size * FLOAT_SIZE;
            println!("(wk) start = {}, offset = {}", start, offset);            
            let wk = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * n_kv_heads * head_size * FLOAT_SIZE;
            println!("(wv) start = {}, offset = {}", start, offset);            
            let wv = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * n_heads * head_size * dim * FLOAT_SIZE;
            println!("(wo) start = {}, offset = {}", start, offset);            
            let wo = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * FLOAT_SIZE;
            println!("(rms_ffn_weight) start = {}, offset = {}", start, offset);            
            let rms_ffn_weight = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * hidden_dim * FLOAT_SIZE;
            println!("(w1) start = {}, offset = {}", start, offset);            
            let w1 = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * hidden_dim * dim * FLOAT_SIZE;
            println!("(w2) start = {}, offset = {}", start, offset);            
            let w2 = &data[start..offset];

            let start = offset;
            let offset = start + n_layers * dim * hidden_dim * FLOAT_SIZE;
            println!("(w3) start = {}, offset = {}", start, offset);            
            let w3 = &data[start..offset];

            let start = offset;
            let offset = start + dim * FLOAT_SIZE;  // Skipping rms_final_weight for now
            println!("(rms_final_weight) start = {}, offset = {}", start, offset);            
            let rms_final_weight = &data[start..offset];

            let start = offset;
            let offset = start + seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_real
            println!("(freq_cis_real_imag) start = {}, offset = {}", start, offset);            
            let start = offset;
            let offset = start + seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_imag
            println!("(wcls) start = {}, offset = {}", start, offset);            

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
                rms_final_weight: slice::from_raw_parts(rms_final_weight.as_ptr() as *const f32, rms_final_weight.len() / FLOAT_SIZE), 
                wcls: wcls,       
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
