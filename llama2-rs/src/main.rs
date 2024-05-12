use memmap2::{Mmap, MmapOptions};
use std::slice;
use std::fs::File;
use std::io;

use std::{mem,fs};
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
        let raw_i32_to_usize = |b: &[u8]| -> io::Result<usize> {
            let num = b.try_into().map_err(
                |_| io::Error::new(ErrorKind::InvalidData, "Invalid byte slice")
            ).map(i32::from_le_bytes)?;
            Ok(num as usize)
        };

        Ok(Config {
            dim: raw_i32_to_usize(&bytes[0..4])?,
            hidden_dim: raw_i32_to_usize(&bytes[4..8])?,
            n_layers: raw_i32_to_usize(&bytes[8..12])?,
            n_heads: raw_i32_to_usize(&bytes[12..16])?,
            n_kv_heads: raw_i32_to_usize(&bytes[16..20])?,
            vocab_size: raw_i32_to_usize(&bytes[20..24])?,
            seq_len: raw_i32_to_usize(&bytes[24..28])?,
        })
    }
}


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
    fn new(data: &'a Mmap, cfg: &Config, max_index: usize) -> Self {

        let head_size = cfg.dim / cfg.n_heads;

        let start_tok_emb_tbl = CONFIG_SIZE_IN_BYTES;
        let end_tok_emb_tbl = start_tok_emb_tbl + cfg.vocab_size * cfg.dim * FLOAT_SIZE;
        // let token_embedding_table = &data[start_tok_emb_tbl..end_tok_emb_tbl];

        let start_rms_att_weight = end_tok_emb_tbl;
        let end_rms_att_weight = start_rms_att_weight + cfg.n_layers * cfg.dim * FLOAT_SIZE;
        // let rms_att_weight = &data[start_rms_att_weight..end_rms_att_weight];

        let start_wq = end_rms_att_weight;
        let end_wq = start_wq + cfg.n_layers * cfg.dim * cfg.n_heads * head_size * FLOAT_SIZE;
        // let wq = &data[start_wq..end_wq];

        let start_wk = end_wq;
        let end_wk = start_wk + cfg.n_layers * cfg.dim * cfg.n_kv_heads * head_size * FLOAT_SIZE;
        // let wk = &data[start_wk..end_wk];

        let start_wv = end_wk;
        let end_wv = start_wv + cfg.n_layers * cfg.dim * cfg.n_kv_heads * head_size * FLOAT_SIZE;
        // let wv = &data[start_wv..end_wv];

        let start_wo = end_wv;
        let end_wo = start_wo + cfg.n_layers * cfg.n_heads * head_size * cfg.dim * FLOAT_SIZE;
        // let wo = &data[start_wo..end_wo];

        let start_rms_ffn_weight = end_wo;
        let end_rms_ffn_weight = start_rms_ffn_weight + cfg.n_layers * cfg.dim * FLOAT_SIZE;
        // let rms_ffn_weight = &data[start_rms_ffn_weight..end_rms_ffn_weight];

        let start_w1 = end_rms_ffn_weight;
        let end_w1 = start_w1 + cfg.n_layers * cfg.dim * cfg.hidden_dim * FLOAT_SIZE;
        // let w1 = &data[start_w1..end_w1];

        let start_w2 = end_w1;
        let end_w2 = start_w2 + cfg.n_layers * cfg.hidden_dim * cfg.dim * FLOAT_SIZE;
        // let w2 = &data[start_w2..end_w2];

        let start_w3 = end_w2;
        let end_w3 = start_w3 + cfg.n_layers * cfg.dim * cfg.hidden_dim * FLOAT_SIZE;
        // let w3 = &data[start_w3..end_w3];

        let start_rms_final_weight = end_w3;
        let end_rms_final_weight = start_rms_final_weight + cfg.dim * FLOAT_SIZE;  // Skipping rms_final_weight for now
        // let rms_final_weight = &data[start_rms_final_weight..end_rms_final_weight];

        let start_freq_cis_real = end_rms_final_weight;
        let end_freq_cis_real = start_freq_cis_real + cfg.seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_real
        let start_freq_cis_img = end_freq_cis_real;
        let end_freq_cis_img = start_freq_cis_img + cfg.seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_imag
        let start_wcls = end_freq_cis_img;
        let end_wcls = max_index;

        let map_bytes_to_f32 = |start_index: usize, end_index: usize| -> &'a [f32] {
            unsafe {
                slice::from_raw_parts(data[start_index..end_index].as_ptr() as *const f32, (end_index - start_index) / FLOAT_SIZE)
            }
        };
        
        let shared_weights = if cfg.vocab_size > 0 { true } else { false };
        let wcls = if shared_weights {
            map_bytes_to_f32(start_tok_emb_tbl, end_tok_emb_tbl)
        } else {
            map_bytes_to_f32(start_wcls, end_wcls)
        };

        TransformerWeights {
            token_embedding_table: map_bytes_to_f32(start_tok_emb_tbl, end_tok_emb_tbl), 
            rms_att_weight: map_bytes_to_f32(start_rms_att_weight, end_rms_att_weight), 
            rms_ffn_weight: map_bytes_to_f32(start_rms_ffn_weight, end_rms_ffn_weight), 
            wq: map_bytes_to_f32(start_wq, end_wq), 
            wk: map_bytes_to_f32(start_wk, end_wk), 
            wv: map_bytes_to_f32(start_wv, end_wv), 
            wo: map_bytes_to_f32(start_wo, end_wo), 
            w1: map_bytes_to_f32(start_w1, end_w1), 
            w2: map_bytes_to_f32(start_w2, end_w2), 
            w3: map_bytes_to_f32(start_w3, end_w3), 
            rms_final_weight: map_bytes_to_f32(start_rms_final_weight, end_rms_final_weight), 
            wcls: wcls,       
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

fn get_file_size(file_path: &str) -> std::io::Result<usize> {
    let metadata = fs::metadata(file_path)?;
    Ok(metadata.len() as usize)
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

    let file_size = get_file_size(&cli.checkpoint_path)?; 
    let file = File::open(cli.checkpoint_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let config_size = mem::size_of::<Config>();
    let config = Config::from_bytes(&mmap)?;
    let transformer_weights = TransformerWeights::new(&mmap, &config, file_size);



    Ok(())
}
