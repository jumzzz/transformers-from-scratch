#### llama2-rs

##### Description:
- We're already late in the party but we will still attempt to replicate Karpathy's llama2.c in Rust


##### Constraints
- Use `memmap2` crate since Karpathy's implementation uses memmap in raw C.
- Use `clap` crate for command-line interface instead of default Rust cli from standard library.

#### Main Parts
- The `Configuration` part - **DONE**
- The `Transformer` part - **TODO**
- The `Tokenizer` part - **TODO**
- The `Sampler` part - **TODO**
- The `Mode` part (Generate/Chat) - **TODO**
