# llama2-rs

##### Description:
- We're already late in the party but we will still attempt to replicate Karpathy's llama2.c in Rust


##### Constraints
- Use `memmap2` crate since Karpathy's implementation uses memmap in raw C.
- Use `clap` crate for command-line interface instead of default Rust cli from standard library.

#### Main Parts
- The `Configuration` part - **First Iteration: DONE**
- Setup debugger for both `llama2.c` and `llama-rs` - **DONE**
- Read about **Virtual Memory** and **mmap** - **TODO**
- The `Transformer` part - **TODO**
- The `Tokenizer` part - **TODO**
- The `Sampler` part - **TODO**
- The `Mode` part (Generate/Chat) - **TODO**


### Parsing Model File

#### Header Parsing for `Config`
First, we fill up the necessary data for the configuration:

```rust
struct Config {
    dim : i32,
    hidden_dim : i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}
```
Which can be derived from the header of `../models/stories15M.bin`
![header](assets/header_models.png)

Now to do these, we need a a way to parse raw bytes to i32. Here's the sample implementation
```rust
impl Config {
    fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 28 {
            return Err(io::Error::new(ErrorKind::Other, "Insufficient bytes for Config"));
        }
        Ok(Config {
            dim: Config::to_i32(&bytes[0..4])?,
            hidden_dim: Config::to_i32(&bytes[4..8])?,
            n_layers: Config::to_i32(&bytes[8..12])?,
            n_heads: Config::to_i32(&bytes[12..16])?,
            n_kv_heads: Config::to_i32(&bytes[16..20])?,
            vocab_size: Config::to_i32(&bytes[20..24])?,
            seq_len: Config::to_i32(&bytes[24..28])?,
        })
    }

    fn to_i32(bytes: &[u8]) -> io::Result<i32> {
        bytes.try_into()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid byte slice"))
            .map(i32::from_ne_bytes)
    }
```

Which allows us to do the following:
```rust
    let file = File::open(cli.checkpoint_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let config_size = mem::size_of::<Config>();
    let config = Config::from_bytes(&mmap)?;
```


### Challenges:

#### Representing Slices to Different Types
- The model weights data which will come from a file will certainly be represented initially as `&[u8]`. Meanwhile, this needs to be represented as `&[f32]`. The main challenge here is how to safely cast `&[u8]` -> `&[f32]`. This should be done such that we can maintain the ownership of the owner of `&[u8]`.  