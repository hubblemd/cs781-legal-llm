import heapq
import json
import os
import re
import time
from collections import Counter, defaultdict

PRETOKENIZE_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)
EOS_TOKEN = "<|endoftext|>"

class LegalBPETokenizer:
    def __init__(self, merges=None, name="legal-bpe"):
        self.name = name
        self.merges = [tuple(m) for m in (merges or [])]
        self._pattern = re.compile(PRETOKENIZE_PATTERN)
        self._build_tables()

    def _build_tables(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eos_token = EOS_TOKEN
        self.eos_token_id = 256 + len(self.merges)
        self.pad_token = self.eos_token
        self.pad_token_id = self.eos_token_id
        self.model_max_length = 10**9
        self._cache = {}

    def __len__(self):
        return 256 + len(self.merges) + 1

    @property
    def vocab_size(self):
        return len(self)

    def pretokenize(self, text):
        return self._pattern.findall(text)

    @classmethod
    def train(cls, texts, vocab_size, name="legal-bpe", min_frequency=2, verbose=True):
        assert 258 <= vocab_size <= 65535, "vocab_size must fit in uint16"
        t0 = time.time()
        pattern = re.compile(PRETOKENIZE_PATTERN)

        word_counts = Counter()
        n_chars = 0
        for text in texts:
            n_chars += len(text)
            word_counts.update(pattern.findall(text))
        if verbose:
            print(f"[BPE] {n_chars:,} characters, {len(word_counts):,} unique words "
                  f"({time.time() - t0:.0f}s)")

        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())
        del word_counts

        pair_counts = defaultdict(int)
        pair_where = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for a, b in zip(w, w[1:]):
                pair_counts[(a, b)] += f
                pair_where[(a, b)].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        n_merges = vocab_size - 257
        merges = []
        while len(merges) < n_merges and heap:
            neg_c, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count <= 0:
                continue
            if -neg_c != count:
                heapq.heappush(heap, (-count, pair))
                continue
            if count < min_frequency:
                if verbose:
                    print(f"[BPE] stopping early: best pair frequency {count} < {min_frequency}")
                break

            new_id = 256 + len(merges)
            merges.append(pair)
            a, b = pair
            changed = set()
            for wi in list(pair_where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                new_w, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                        new_w.append(new_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[wi] = new_w
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] += f
                    pair_where[p].add(wi)
                    changed.add(p)
            del pair_where[pair]
            pair_counts.pop(pair, None)
            changed.discard(pair)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
                    pair_where.pop(p, None)
            if verbose and (len(merges) % 2000 == 0):
                print(f"[BPE] {len(merges):,}/{n_merges:,} merges ({time.time() - t0:.0f}s)")

        tok = cls(merges, name=name)
        if verbose:
            print(f"[BPE] done: vocab size {len(tok):,} ({time.time() - t0:.0f}s)")
        return tok

    def _encode_word(self, word):
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        ids = list(word.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            a, b = best_pair
            new_id = 256 + best_rank
            merged, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def encode(self, text, add_special_tokens=False):
        out = []
        for word in self._pattern.findall(text):
            out.extend(self._encode_word(word))
        return out

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts)}
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        parts = []
        for i in ids:
            i = int(i)
            if i == self.eos_token_id:
                if not skip_special_tokens:
                    parts.append(self.eos_token.encode("utf-8"))
                continue
            b = self.vocab.get(i)
            if b is not None:
                parts.append(b)
        return b"".join(parts).decode("utf-8", errors="replace")

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i].decode("utf-8", errors="replace") if i in self.vocab else self.eos_token
                for i in ids]

    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "legal_bpe_tokenizer.json")
        with open(path, "w") as f:
            json.dump({
                "type": "byte-level-bpe",
                "name": self.name,
                "pattern": PRETOKENIZE_PATTERN,
                "eos_token": self.eos_token,
                "eos_token_id": self.eos_token_id,
                "vocab_size": len(self),
                "merges": [list(m) for m in self.merges],
            }, f)
        return path

    @classmethod
    def from_pretrained(cls, directory):
        path = directory if directory.endswith(".json") else os.path.join(directory, "legal_bpe_tokenizer.json")
        with open(path) as f:
            data = json.load(f)
        assert data["pattern"] == PRETOKENIZE_PATTERN, "tokenizer was saved with a different pre-tokenisation pattern"
        tok = cls(data["merges"], name=data.get("name", "legal-bpe"))
        assert tok.eos_token_id == data["eos_token_id"]
        return tok
