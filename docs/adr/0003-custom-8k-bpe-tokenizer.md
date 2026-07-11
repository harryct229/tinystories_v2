# Custom 8k-vocab BPE tokenizer instead of reusing GPT-2's

We train our own byte-level BPE tokenizer (~8k vocab) on TF1-EN-3M rather than
reusing the standard GPT-2 tokenizer (50k vocab). At d=512 with tied
embeddings, a 50k vocab costs ≈26M parameters — the entire ~25M budget — while
the corpus only contains ~11k unique words. An 8k vocab keeps embeddings at
≈4M params (~16% of budget), leaving capacity for transformer layers. The
tokenizer is frozen once Pretraining starts; every checkpoint depends on it.
