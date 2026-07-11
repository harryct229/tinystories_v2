# Llama-style architecture for the from-scratch model

We chose a Llama-style decoder-only stack (pre-norm RMSNorm, RoPE, SwiGLU
feed-forward, no biases, tied embeddings, standard MHA) over the GPT-2-style
stack (LayerNorm, learned absolute positions, GELU, biases) that the original
TinyStories work used. Rationale: every component has a citable justification
(RMSNorm: Zhang & Sennrich 2019; RoPE: Su et al. 2021; SwiGLU: Shazeer 2020),
it is what modern small-LM work (Llama, Qwen, SmolLM) converged on, and the
course explicitly grades justification of layer choices — defending the 2023
consensus stack is a stronger academic story than defending 2019 choices the
field has replaced. Target scale ≈ 25M params (d=512, 8 layers, 8 heads).
Pretraining compute is sunk once spent, so this is effectively irreversible.
