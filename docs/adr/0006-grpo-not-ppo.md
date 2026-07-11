# GRPO instead of PPO for the RL stage

Although we frame stage 3 as a faithful InstructGPT reproduction (ADR-0004),
we run GRPO rather than InstructGPT's original PPO. GRPO replaces the learned
value network with the group-mean reward baseline (G completions per Slot
Prompt), removing the value model, its loss, and its hyperparameters. At 25M
policy scale the extra rollouts are nearly free while the value network is
pure added instability, and GRPO is current RLHF practice (DeepSeek-R1 era).
The report should state this substitution explicitly and justify why the
value network became unnecessary.
