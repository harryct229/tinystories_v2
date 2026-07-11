# Full InstructGPT recipe (Reward Model + RL) instead of DPO

For the RLAIF stage we chose the faithful InstructGPT pipeline — distill Judge
(Qwen) labels into a small Reward Model, then run policy-gradient RL against
it — over the simpler DPO alternative. Rationale: the course brief explicitly
names Reinforcement Learning, DPO performs none (it is a closed-form
derivation of the RLHF objective), and at ~25M policy scale true RL is
actually feasible on a T4 because the Judge is only needed offline for RM
label collection, never in the training loop. Accepted cost: three extra
moving parts (RM training, RM validation, online RL loop) and higher schedule
risk than DPO.
