# app/bigram_model.py
from collections import Counter
import numpy as np
import re
from typing import List

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def _tokenize(text: str) -> List[str]:
    # Class note style: simple, lowercase, alphanum + apostrophes
    return _WORD_RE.findall(text.lower())

class BigramModel:
    """
    Bigram text generator aligned with Module 2 Word Sampling:
      - Split corpus into sentences
      - Add BOS/EOS tokens
      - Build bigram counts
      - Row-normalize to get P(next | current)
      - Sample sequentially to generate text
    """
    def __init__(self, corpus: List[str], bos_token: str = "<s>", eos_token: str = "</s>"):
        self.bos = bos_token
        self.eos = eos_token
        self.vocab: List[str] = []
        self.word_to_idx = {}
        self._P = None  # row-stochastic matrix
        self._fit(corpus)

    def _fit(self, corpus: List[str]) -> None:
        # 1) Sentence split + tokenize + add BOS/EOS (per class note)
        sentences: List[List[str]] = []
        for doc in corpus:
            for raw in re.split(r"[.!?]+", doc):
                toks = _tokenize(raw)
                if toks:
                    sentences.append([self.bos] + toks + [self.eos])

        # 2) Vocab from observed tokens
        vocab_counter = Counter()
        for s in sentences:
            vocab_counter.update(s)
        self.vocab = sorted(vocab_counter.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)

        # 3) Bigram count matrix
        M = np.zeros((V, V), dtype=np.float64)
        for s in sentences:
            for a, b in zip(s[:-1], s[1:]):      # classic bigram pairing
                i = self.word_to_idx[a]
                j = self.word_to_idx[b]
                M[i, j] += 1.0

        # 4) Row-normalize to probabilities P(next | current)
        row_sums = M.sum(axis=1, keepdims=True)
        # Avoid division by zero (rows with no outgoing edges)
        row_sums[row_sums == 0] = 1.0
        self._P = M / row_sums

    def generate_text(self, start_word: str, length: int = 10) -> str:
        """
        Sequentially sample next tokens using P(next | current).
        - If start_word not in vocab, fall back to BOS to start a sentence.
        - Stops early on EOS.
        """
        if length <= 0:
            return ""

        w0 = start_word.lower()
        if w0 not in self.word_to_idx:
            w0 = self.bos
        i = self.word_to_idx[w0]

        out: List[str] = []
        for _ in range(length):
            probs = self._P[i]
            j = int(np.random.choice(len(self.vocab), p=probs))
            nxt = self.vocab[j]
            if nxt == self.eos:
                break
            if nxt not in (self.bos, self.eos):
                out.append(nxt)
            i = j

        return " ".join(out).strip()