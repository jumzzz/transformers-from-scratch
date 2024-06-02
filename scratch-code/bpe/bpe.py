from transformers import AutoTokenizer
from collections import defaultdict
import json

tokenizer = AutoTokenizer.from_pretrained("gpt2") 

def print_json(data):
    print(json.dumps(data, indent=2))

def print_line_by_line(data, limit=-1):
    if limit == -1:
        assigned_limit = len(data)
    else:
        assigned_limit = limit 

    for i, data in enumerate(data.items(), start=0):
        key, val = data
        if i >= assigned_limit:
            break
        print(f"{key} = {val}")
    print("")

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
    
    for word in new_words:
        word_freqs[word] += 1

alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()
vocab = ["<|endoftext|>"] + alphabet.copy()

print("")
splits = {word: [c for c in word] for word in word_freqs.keys()}

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    print("computing pairs:")
    for word, freq in word_freqs.items():
        split = splits[word]
        print(f"word = {word}, splits[{word}] = {split}")
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    
    print("")
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)

print("pair_freqs:")
print_line_by_line(pair_freqs)


best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)
