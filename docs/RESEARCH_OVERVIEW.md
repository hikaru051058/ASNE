# Research Overview

## Problem Statement

Neural encoding models predict cortical responses from stimulus inputs such as images, video, or text. Most workflows evaluate a mapping from stimulus features to predicted cortical response patterns under a fixed model condition. ASNE asks whether a frozen neural encoding model can be used to study controlled changes in predicted cortical responses when the same stimulus is paired with a defined steering condition.

The goal is not to infer a person's true thoughts or emotions. The goal is to evaluate whether model-side conditioning produces measurable, reproducible response deltas in predicted cortical-response space.

## Research Hypothesis

For the same stimulus, controllable steering conditions can produce distinguishable and interpretable changes in predicted cortical response patterns from a frozen neural encoding model.

Operationally, ASNE tests whether:

- Response deltas differ from neutral baseline predictions.
- Response deltas are consistent across related stimuli.
- ROI-level shifts follow interpretable patterns.
- Steering conditions are separable in predicted response space.
- Steering effects remain distinguishable from prompt-only baselines.

## Novelty Claim

ASNE combines neural encoding evaluation with activation or prompt steering as an experimental control. Instead of only asking how a stimulus maps to a predicted cortical response, ASNE asks how that mapping changes under explicit steering conditions:

```text
traditional neural encoding: response = f(stimulus)
ASNE: response = f(stimulus, steering_state)
```

The novelty is in treating steering as a controlled variable for cortical-response delta analysis around a frozen upstream neural-response model.

## Relation to Neural Encoding and Activation Steering

Neural encoding models estimate response patterns associated with external stimuli. Activation steering methods modify or condition internal model representations to bias generated or predicted behavior. ASNE connects these ideas by using steering conditions to modulate the prediction process while preserving the upstream model as a fixed reference.

Possible steering implementations include:

- Prompt prefixes.
- Soft prompts.
- Activation vectors.
- Representation steering layers.
- Future LoRA or adapter steering.

TODO: Determine which steering mechanisms are compatible with the actual TRIBE v2 API and model architecture.

## Why Context and State Conditioning Matters

Human cortical responses are shaped by attention, task context, expectation, memory, and other contextual factors. ASNE does not claim to simulate those mechanisms directly. Instead, it provides a computational framework for asking whether model-side state conditioning creates structured changes in predicted cortical-response outputs.

This matters because a state-conditioned encoding framework could help researchers inspect how latent context variables alter model-predicted cortical representations, identify which ROIs are most sensitive to steering conditions, and compare different steering mechanisms under shared stimuli.

## Safe Scientific Interpretation

ASNE outputs are model predictions. A response delta should be interpreted as a change in predicted cortical response under a defined computational condition, not as detected emotion, diagnosis, belief, intent, or consciousness.

Scientific claims should be limited to the tested model, stimuli, steering definitions, and evaluation procedure. Claims about empirical neuroscience require independent validation against human neural measurements and careful citation of prior work.

## Possible Evaluation Metrics

- Delta magnitude: Norm or average absolute difference between neutral and steered predicted cortical responses.
- ROI-level response shift: Region-wise summary of response deltas and rank ordering of affected ROIs.
- Clustering and separability: Whether steering conditions form distinguishable groups in predicted response space.
- Consistency across stimuli: Whether a steering condition produces stable response-delta patterns over multiple stimuli.
- Comparison against prompt-only baselines: Whether deeper steering mechanisms produce effects beyond simple text-prefix conditioning.

## Limitations

- Predicted cortical responses are model outputs, not direct human measurements.
- Steering conditions are computational controls, not verified mental or emotional states.
- Results may depend strongly on the upstream model, training data, stimulus set, ROI definitions, and steering implementation.
- Frozen upstream models may contain biases or artifacts inherited from their data and training objectives.
- Interpretability of response deltas is limited without empirical validation.
- TODO: Add model-specific limitations after reviewing TRIBE v2 documentation, output format, and licensing constraints.
