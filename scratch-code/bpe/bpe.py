from transformers import AutoTokenizer
from collections import defaultdict
import json

tokenizer = AutoTokenizer.from_pretrained("gpt2") 

def print_json(data):
    print(json.dumps(data, indent=2))


corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

word_freqs = defaultdict(int)

total_corpus = len(corpus)

for cnt, text in enumerate(corpus, start=1):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)

    new_words = [word for word, offset in words_with_offsets]
    
    print(f"[{cnt}/{total_corpus}] words_with_offsets = {words_with_offsets}")
    print(f"[{cnt}/{total_corpus}] new_words = {new_words}")
    print("")
    for word in new_words:
        word_freqs[word] += 1

word_freqs_dict = dict(word_freqs)
print_json(word_freqs_dict)


