from adapters import run_train_bpe

input_path = "/Users/cyy1/cs336/minimal-language-model-training/data/TinyStoriesV2-GPT4-train.txt"
special_token = ["<|endoftext|>"]

run_train_bpe(input_path, 100, special_token)
