# 11 — Reference-free metrics library

Status: complete

## Parent

`.scratch/tinystories-v2-pipeline/PRD.md` (extracted from issue 07 to unblock
parallel work)

## What to build

The metric implementations the report compares against the dataset paper's
tables, as a pure library with no model, GPU, or network dependencies —
consumed by the eval suite (issue 07) and by GRPO's diversity monitoring
(issue 06).

- **Self-BLEU** (diversity: lower = more diverse) over a set of fables
- **Distinct-n** (lexical richness; n=1 at minimum, matching the paper)
- **Flesch Reading Ease** (age fit)
- **Perplexity helper**: takes a model and tokenized held-out text, returns
  perplexity — usable with any checkpoint, including toy test models

## Acceptance criteria

- [x] Each text metric is tested against hand-computed values on tiny inputs (test_distinct_1_hand_computed, test_self_bleu_partial_overlap_hand_computed, test_flesch_single_sentence_hand_computed, and siblings)
- [x] Edge cases covered: empty set, single fable, identical fables (Self-BLEU extreme), degenerate short texts (test_distinct_rejects_empty_set, test_distinct_works_on_a_single_fable, test_self_bleu_rejects_fewer_than_two_fables, test_self_bleu_identical_fables_is_maximally_redundant, test_self_bleu_single_token_fables, test_short_fables_contribute_zero_ngrams)
- [x] Perplexity helper matches a hand-rolled loss computation on the fixture with a toy model, on CPU (test_matches_hand_rolled_nll_on_fixture; test_uniform_model_perplexity_is_vocab_size pins the closed-form case)
- [x] Metrics accept plain lists of strings — no coupling to stage artifacts or model classes (except the perplexity helper's model argument); test_import_never_eagerly_pulls_torch guards that neither module needs torch to import
- [x] Deterministic given the same inputs (test_self_bleu_sampling_is_seeded_and_deterministic, test_deterministic_across_calls); Self-BLEU sampling is seeded via explicit sample_size/seed parameters that consuming stages wire to their configs

## Blocked by

None — can start immediately (issue 01 is complete).
