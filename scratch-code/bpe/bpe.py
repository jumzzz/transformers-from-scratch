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
    # print("computing pairs:")
    for word, freq in word_freqs.items():
        split = splits[word]
        # print(f"word = {word}, splits[{word}] = {split}")
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    
    # print("")
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)

print("pair_freqs:")
print_line_by_line(pair_freqs, 5)


best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print("most frequent pair:")
print(f"{best_pair} = {max_freq}")

merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

splits = merge_pair("Ġ", "t", splits)

print("")
print("initial splits: ")
print_line_by_line(splits, -1)

vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(best_pair[0], best_pair[1], splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print("merges:")
print_line_by_line(merges, -1)

print("final splits:")
print_line_by_line(splits)


def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

print("Sample tokenization: this is not a tokenization.")
print(tokenize("this is not a tokenization"))
