from collections.abc import Iterable
from typing import Iterator
import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. 
        This function should accept the following parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        import json

        # Load vocab: {"id": "hex_string"} -> {int: bytes}
        with open(vocab_filepath, "r") as f:
            raw_vocab = json.load(f)
        vocab = {}
        for k, v in raw_vocab.items():
            vocab[int(k)] = bytes.fromhex(v)

        # Load merges: [["hex1", "hex2"], ...] -> list[tuple[bytes, bytes]]
        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)
        merges = []
        for a, b in raw_merges:
            merges.append((bytes.fromhex(a), bytes.fromhex(b)))

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # Build reverse lookup: bytes -> token ID
        bytes_to_id = {v: k for k, v in self.vocab.items()}
        # Build merge priority: (a, b) -> rank (lower = higher priority)
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}

        encoded_list = []

        # Handle special tokens first: split text around them
        if self.special_tokens:
            special_pat = "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            parts = re.split(special_pat, text)
        else:
            parts = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                encoded_list.append(bytes_to_id[part.encode("utf-8")])
                continue

            # Pre-tokenize this part
            pre_tokens = re.findall(PAT, part)
            for word in pre_tokens:
                token_seq = [bytes([b]) for b in word.encode("utf-8")]

                # TODO(human): apply BPE merges to token_seq using merge_rank
                # Result should be a list of bytes objects after all merges applied
                while True:
                    best_rank = float('inf')                                                                                                                                              
                    best_pair = None                                                                                                                                                      
                    for i in range(len(token_seq) - 1):                                                                                                                                   
                        pair = (token_seq[i], token_seq[i+1])                                                                                                                             
                        rank = merge_rank.get(pair, float('inf'))                                                                                                                         
                        if rank < best_rank:                                                                                                                                              
                            best_rank = rank                                                                                                                                              
                            best_pair = pair

                    
                    if best_pair not in merge_rank:                                                                                                                                   
                        break  

                    new_seq = []
                    i = 0                                                                                                                                                             
                    while i < len(token_seq):
                        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:                                                                                    
                            new_seq.append(token_seq[i] + token_seq[i+1])  # 字节拼接                                                                                                 
                            i += 2                                                                                                                                                    
                        else:                                                                                                                                                         
                            new_seq.append(token_seq[i])                                                                                                                              
                            i += 1
                    token_seq = new_seq
                    
                encoded_list.extend(bytes_to_id[t] for t in token_seq)

        return encoded_list


    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            # encode 每一行，逐个 yield token id
            ids = self.encode(text)
            for token_id in ids:
                yield token_id


    def decode(self, ids: list[int]) -> str: 
        """
        Decode a sequence of token IDs into text.
        """                                                                                                                    
        bytes_list = []                                                                                                                                                   
        for i in ids:                                                                                                                                                     
            bytes_list.append(self.vocab[i])                                                                                                                              
                                                                                                                                                                                                                                                                                    
        all_bytes = b"".join(bytes_list)
                                                                                                                                                                            
        return all_bytes.decode("utf-8", errors="replace")  


