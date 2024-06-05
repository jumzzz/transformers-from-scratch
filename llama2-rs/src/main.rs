use memmap2::{Mmap, MmapOptions};
use std::slice;
use std::fs::File;
use std::io;

use std::{mem,fs};
use clap::{Parser,ValueEnum};
use std::io::{ErrorKind, Read};

use std::string::FromUtf8Error;

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

struct IndexRange {
    start: usize,
    end: usize,
}

impl IndexRange {
    fn new(start: usize, end: usize) -> IndexRange {
        IndexRange {
            start: start,
            end: end,
        }
    }
}

struct TransformerIndexRange {
    token_embedding_table   : IndexRange,
    rms_att_weight          : IndexRange, 
    rms_ffn_weight          : IndexRange,
    wq                      : IndexRange,
    wk                      : IndexRange,
    wv                      : IndexRange,
    wo                      : IndexRange,
    w1                      : IndexRange,
    w2                      : IndexRange,
    w3                      : IndexRange,
    rms_final_weight        : IndexRange,
    wcls                    : IndexRange,
}

impl TransformerIndexRange {
    fn new(cfg: &Config, max_index: usize) -> TransformerIndexRange {
        let head_size = cfg.dim / cfg.n_heads;

        let start_tok_emb_tbl = CONFIG_SIZE_IN_BYTES;
        let end_tok_emb_tbl = start_tok_emb_tbl + cfg.vocab_size * cfg.dim * FLOAT_SIZE;

        let start_rms_att_weight = end_tok_emb_tbl;
        let end_rms_att_weight = start_rms_att_weight + cfg.n_layers * cfg.dim * FLOAT_SIZE;

        let start_wq = end_rms_att_weight;
        let end_wq = start_wq + cfg.n_layers * cfg.dim * cfg.n_heads * head_size * FLOAT_SIZE;

        let start_wk = end_wq;
        let end_wk = start_wk + cfg.n_layers * cfg.dim * cfg.n_kv_heads * head_size * FLOAT_SIZE;

        let start_wv = end_wk;
        let end_wv = start_wv + cfg.n_layers * cfg.dim * cfg.n_kv_heads * head_size * FLOAT_SIZE;

        let start_wo = end_wv;
        let end_wo = start_wo + cfg.n_layers * cfg.n_heads * head_size * cfg.dim * FLOAT_SIZE;

        let start_rms_ffn_weight = end_wo;
        let end_rms_ffn_weight = start_rms_ffn_weight + cfg.n_layers * cfg.dim * FLOAT_SIZE;

        let start_w1 = end_rms_ffn_weight;
        let end_w1 = start_w1 + cfg.n_layers * cfg.dim * cfg.hidden_dim * FLOAT_SIZE;

        let start_w2 = end_w1;
        let end_w2 = start_w2 + cfg.n_layers * cfg.hidden_dim * cfg.dim * FLOAT_SIZE;

        let start_w3 = end_w2;
        let end_w3 = start_w3 + cfg.n_layers * cfg.dim * cfg.hidden_dim * FLOAT_SIZE;

        let start_rms_final_weight = end_w3;
        let end_rms_final_weight = start_rms_final_weight + cfg.dim * FLOAT_SIZE;  // Skipping rms_final_weight for now

        let start_freq_cis_real = end_rms_final_weight;
        let end_freq_cis_real = start_freq_cis_real + cfg.seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_real
        let start_freq_cis_img = end_freq_cis_real;
        let end_freq_cis_img = start_freq_cis_img + cfg.seq_len * head_size * FLOAT_SIZE / 2; // Skip freq_cis_imag
        let start_wcls = end_freq_cis_img;
        let end_wcls = max_index;

        TransformerIndexRange {
            token_embedding_table:  IndexRange::new(start_tok_emb_tbl, end_tok_emb_tbl), 
            rms_att_weight:         IndexRange::new(start_rms_att_weight, end_rms_att_weight), 
            rms_ffn_weight:         IndexRange::new(start_rms_ffn_weight, end_rms_ffn_weight),
            wq:                     IndexRange::new(start_wq, end_wq),
            wk:                     IndexRange::new(start_wk, end_wk),
            wv:                     IndexRange::new(start_wv, end_wv),
            wo:                     IndexRange::new(start_wo, end_wo),
            w1:                     IndexRange::new(start_w1, end_w1),
            w2:                     IndexRange::new(start_w2, end_w2),
            w3:                     IndexRange::new(start_w3, end_w3),
            rms_final_weight:       IndexRange::new(start_rms_final_weight, end_rms_final_weight),
            wcls:                   IndexRange::new(start_wcls, end_wcls),
        } 
    }
}

struct TransformerWeights <'a>{
    token_embedding_table:  &'a [f32], 
    rms_att_weight:         &'a [f32],
    rms_ffn_weight:         &'a [f32],
    wq:                     &'a [f32],
    wk:                     &'a [f32],
    wv:                     &'a [f32],
    wo:                     &'a [f32],
    w1:                     &'a [f32],
    w2:                     &'a [f32],
    w3:                     &'a [f32],
    rms_final_weight:       &'a [f32],
    wcls:                   &'a [f32],
}

impl<'a> TransformerWeights <'a> {
    fn load(data: &'a Mmap, cfg: &Config, max_index: usize) -> Self {
        let index_ranges = TransformerIndexRange::new(cfg, max_index);
        let map_bytes_to_f32 = |index_range: &IndexRange| -> &'a [f32] {
            unsafe {
                let start_index = index_range.start;
                let end_index = index_range.end;
                slice::from_raw_parts(data[start_index..end_index].as_ptr() as *const f32, (end_index - start_index) / FLOAT_SIZE)
            }
        };
        
        let shared_weights = if cfg.vocab_size > 0 { true } else { false };
        let wcls = if shared_weights {
            map_bytes_to_f32(&index_ranges.token_embedding_table)
        } else {
            map_bytes_to_f32(&index_ranges.wcls)
        };

        TransformerWeights {
            token_embedding_table:  map_bytes_to_f32(&index_ranges.token_embedding_table), 
            rms_att_weight:         map_bytes_to_f32(&index_ranges.rms_att_weight), 
            rms_ffn_weight:         map_bytes_to_f32(&index_ranges.rms_ffn_weight), 
            wq:                     map_bytes_to_f32(&index_ranges.wq), 
            wk:                     map_bytes_to_f32(&index_ranges.wk), 
            wv:                     map_bytes_to_f32(&index_ranges.wv), 
            wo:                     map_bytes_to_f32(&index_ranges.wo), 
            w1:                     map_bytes_to_f32(&index_ranges.w1), 
            w2:                     map_bytes_to_f32(&index_ranges.w2), 
            w3:                     map_bytes_to_f32(&index_ranges.w3), 
            rms_final_weight:       map_bytes_to_f32(&index_ranges.rms_final_weight), 
            wcls:                   wcls,       
        } 
    }
}


struct Transformer<'a> {
    config: Config,
    transformer_weights: TransformerWeights<'a>,
    
}

impl<'a> Transformer<'a> {
    fn load(cp_mmap: &'a Mmap, cp_file_size: usize) -> io::Result<Self> {
        
        let config = Config::from_bytes(&cp_mmap)?;
        let transformer_weights = TransformerWeights::load(&cp_mmap, &config, cp_file_size);

        Ok(
            Transformer {
                config,
                transformer_weights,
            }
        )
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}



fn read_bytes_to_float(file_io: &mut File) -> io::Result<f32> {
    let mut buffer = [0u8; mem::size_of::<f32>()];
    file_io.read_exact(&mut buffer)?;
    let f32_from_bytes = f32::from_le_bytes(buffer);

    Ok(f32_from_bytes)
}

fn read_bytes_to_str(file_io: &mut File, str_len: usize) -> io::Result<String> {
    let mut buffer = vec![0u8; str_len];
    file_io.read_exact(&mut buffer)?;
    Ok(String::from_utf8(buffer).unwrap())
}

fn read_int_bytes_to_usize(file_io: &mut File)  -> io::Result<usize> {
    let mut buffer = [0u8; mem::size_of::<i32>()];
    file_io.read_exact(&mut buffer)?;
    let i32_from_bytes = i32::from_le_bytes(buffer);
    
    Ok(i32_from_bytes as usize)
}


struct Tokenizer {
    vocabs: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_size: usize,
    max_token_length: usize,
    byte_pieces: Vec<char>,
}


impl Tokenizer {
    fn load(tokenizer_path: &str, vocab_size: usize) -> io::Result<Self> {
        let mut byte_pieces = vec!['\0'; 512];

        for i in 0..256 { 
            byte_pieces[i * 2] = (i as u8) as char; 
        }

        let vocab_size = vocab_size;
        
        let mut tok_file = File::open(tokenizer_path)?;
        let mut vocab_scores = Vec::with_capacity(vocab_size);
        let mut vocabs = Vec::with_capacity(vocab_size);

        let max_token_length = read_int_bytes_to_usize(&mut tok_file)?;
        println!("max_token_length = {}", max_token_length);
        
        for i in 0..vocab_size {
            let vocab_score = read_bytes_to_float(&mut tok_file)?;
            let vocab_len = read_int_bytes_to_usize(&mut tok_file)?;
            let vocab = read_bytes_to_str(&mut tok_file, vocab_len)?;
            
            println!("[{}/{}] vocab = {}, vocab_len = {}, vocab_score = {}", &i , &vocab_size ,&vocab, &vocab_len, &vocab_score);

            vocab_scores.push(vocab_score);
            vocabs.push(vocab);
        }

        Ok( 
            Tokenizer {
                vocabs,
                vocab_scores,
                vocab_size,
                max_token_length,
                byte_pieces,
            }
        )
    }
}

struct TokenIndex<'a> {
    vocab: &'a String,
    vocab_id: u32,
}

type SortedVocab<'a> = Vec<TokenIndex<'a>>;

trait SortVocab<'a> {
    fn build_sorted_vocab(&mut self, tokenizer: &'a Tokenizer);
}

impl<'a> SortVocab<'a> for SortedVocab<'a> {
    fn build_sorted_vocab(&mut self, tokenizer: &'a Tokenizer) {
        for (vocab_id, vocab) in (0..).zip(&tokenizer.vocabs) {
            self.push(TokenIndex { vocab: &vocab, vocab_id: vocab_id as u32 });
        }
    } 
}

struct ProbIndex {
    prob: f32,
    index: usize,
}

struct Sampler {
    vocab_size: usize,
    probindex: Vec<ProbIndex>,
    temperature: f32,
    topp: f32,
    rng_state: u64, 
}

impl Sampler {
    fn build(vocab_size: usize, temperature: f32, topp: f32, rng_state: u64) -> io::Result<Self> {
        // TODO: interior mutability later. For now, let's make the whole Sampler as mutable. 
        let probindex = Vec::<ProbIndex>::with_capacity(vocab_size);
        Ok(
            Sampler {
                vocab_size,
                probindex,
                temperature,
                topp,
                rng_state,
            }
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    Generate,
    Chat,
}

fn encode(tokenizer: &Tokenizer, text: &str, bos: u8, eos: u8, tokens: u32, n_tokens: &Vec<u32>) {

    if text.len() == 0 {
        return; 
    }

}

/*
    Problem #1: Do we need to keep sorting the tokenizer.vocabs? 
    - By inspection, `qsort` is just being called when `t->sorted_vocab` is uninitialized.
    - This begs the question on what's stopping us from performing sort with tokenizer.vocabs directly.

    Problem #2: How exactly tokenizer->sorted_vocabs are being used?

    -   `tokenizer->sorted_vocab` is used w.r.t. `str_lookup(str_buffer, t->sorted_vocab, t->vocab_size)`
    -  `str_lookup` returns an `id`. This might be the numeric representation of the token itself.

    Mapping the calls:
    
    M1: qsort calls
    - On probindex (For top-p sampling)
    - On t->sorted_vocab (For tokenization)

    M2: t->vocab references
    - On build_tokenizer function
    - On free_tokenizer function
    - On getting `char *piece = t->vocab[token]` at `decode` function (Watch out for this)
    - When being sorted with a `t->sorted_vocab` reference.
    

    M3: t->sorted_vocab references 

 
*/
fn generate(transformer: &Transformer, tokenizer: &Tokenizer, sampler: &Sampler, prompt: &str, steps: usize) {
    if prompt.len() == 0 {
        return;
    }

    let _num_prompt_tokens: usize = 0;
    let prompt_tokens: Vec<u32> = Vec::with_capacity(prompt.len() + 3 );

    // From Karpathy
    //encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);



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

    let cp_file_size = get_file_size(&cli.checkpoint_path)?; 
    let cp_file = File::open(cli.checkpoint_path)?;
    let cp_mmap = unsafe { MmapOptions::new().map(&cp_file)? };

    let transformer = Transformer::load(&cp_mmap, cp_file_size)?;
    let vocab_size = transformer.vocab_size();
    
    let _tok_file_size = get_file_size(&cli.tokenizer_path)?;
    let _tok_file = File::open(&cli.tokenizer_path)?;

    let tokenizer = Tokenizer::load(&cli.tokenizer_path, vocab_size)?;
    let mut sorted_tokenizer = SortedVocab::with_capacity(tokenizer.vocab_size);
    sorted_tokenizer.build_sorted_vocab(&tokenizer);

    let _sampler = Sampler::build(vocab_size, cli.temperature, cli.topp, cli.rng_seed)?;
    
    match cli.mode {
        Mode::Generate  => { println!("Generate Mode!"); },
        Mode::Chat      => { println!("Chat Mode!"); },
    }

    Ok(())
}
