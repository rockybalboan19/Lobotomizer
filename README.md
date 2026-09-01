# Brain Surgeon

Brain Surgeon is a research prototype for self-probe-driven domain pruning of Mixture-of-Experts (MoE) language models. The workflow generates domain probes, captures the model's own answers for calibration, logs router behavior, ranks experts using the router's observable gating signal, prunes a daughter model, and compares parent vs. daughter on a held-out benchmark. This is intended to support a focused experimental workflow, not a proven production model-compression method.

## What is reused vs. original

This project reuses Hugging Face Transformers for model loading, tokenization, and generation. Torch-Pruning is not used in the final published version of this repo and is therefore not required here.

The original logic in this repository is the Brain Surgeon pipeline itself: live MoE adapter detection, domain self-probing, router monitoring, gating-score-based expert ranking, daughter-model creation, and benchmark reporting.

## Methodology note

This version intentionally does not claim to implement the full EASY-EP formulation.
- self-probe generation does treat the model's own answer as calibration metadata, which follows the input+output calibration finding from the paper
- router monitoring records the actual gate weights that are observable at the router hook, but it does not compute the expert-output L2 norm or the residual-stream shift, because those require hooks on the actual expert outputs and the MoE block's hidden-state before/after pass
- expert selection ranks by the observable gating-score baseline rather than labeling the result as the paper's full contribution formula
- router recalibration is kept as a lightweight slice of the router after pruning, but it is not presented as a full implementation of the paper's complete scoring method
- the project remains single-model / single-domain / single-run by design while keeping the underlying workflow general enough to swap models or domains

## Setup

1. Create a Python environment.
2. Install the project dependencies:

   python setup_dependencies.py

   or, if you prefer the raw requirements file:

   pip install -r requirements.txt

3. Make sure you have a local Hugging Face model directory already downloaded on disk.

## Usage

Run the CLI from the project root:

python brain_surgeon.py

It will ask for:
- the local model path
- a plain-English domain description
- a prune size choice (light / medium / aggressive)

Then it will:
- validate that the model is MoE-style
- generate a calibration set and a held-out set via self-probing
- inspect the live router/expert structure and print an adapter checkpoint
- log expert activations and weighted routing signals over the calibration set
- select experts using a weighted score rather than raw activation frequency
- create a daughter model with fewer experts
- print sample daughter outputs
- evaluate parent vs. daughter on the held-out set
- summarize the results and highlight where the daughter regressed

## Outputs

The tool saves intermediate artifacts such as:
- calibration_questions.json
- heldout_questions.json
- activation_stats.json
- daughter_model/

## Status

This is a research prototype and an experimental workflow rather than a finished, benchmark-validated product. The underlying hypothesis is still being evaluated — the core question is whether self-probe-driven pruning can preserve domain performance while reducing a model's expert count without a heavy recalibration step. This code is built to help test that question, not to claim a proven result.
