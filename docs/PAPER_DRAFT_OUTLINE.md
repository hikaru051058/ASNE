# Paper Draft Outline

## Possible Title

ASNE: Activation-Steered Neural Encoding for State-Conditioned Cortical Response Analysis

## Abstract Draft

Neural encoding models predict cortical responses from sensory or semantic stimuli, typically under a fixed model condition. We introduce ASNE, Activation-Steered Neural Encoding, an experimental framework for evaluating whether controllable steering conditions can alter predicted cortical-response patterns in frozen neural-response models. ASNE compares baseline and steered predictions for the same stimulus, computes response deltas, and summarizes effects at whole-response and ROI levels. The framework is designed for cautious computational research: steering conditions are treated as experimental controls, not as detected emotions or psychological states. We outline an initial evaluation using one short video, six steering conditions, response-delta maps, ROI ranking, and clustering analysis to test whether state-conditioned predictions are reproducible, distinguishable, and interpretable.

## Section Outline

1. Introduction
2. Related Work
3. Method
4. Experiments
5. Results
6. Discussion
7. Limitations
8. Ethics

## 1. Introduction

- Present the neural encoding setup and the motivation for state-conditioned prediction.
- Define ASNE as `response = f(stimulus, steering_state)`.
- State the main research question around reproducible, distinguishable, and interpretable response deltas.
- Clarify that ASNE does not perform mind reading, emotion detection, or psychological diagnosis.

## 2. Related Work

- Neural encoding models and cortical-response prediction.
- Foundation models for neural response prediction, including TRIBE v2 or similar systems.
- Prompt conditioning, activation steering, soft prompts, and representation steering.
- Context effects, attention, and task conditioning in neuroscience.
- TODO: Add precise citations after literature review.

## 3. Method

- Describe frozen upstream neural-response model usage.
- Define steering conditions and baseline condition.
- Explain baseline versus steered inference.
- Define response delta computation.
- Define ROI-level analysis and condition separability metrics.
- TODO: Fill in TRIBE v2-specific implementation details after API review.

## 4. Experiments

First experiment design:

- Use one short video stimulus.
- Run six steering conditions: neutral, threat, nostalgia, social judgment, curiosity, and cognitive load.
- Compare baseline versus steered predicted cortical responses.
- Compute response delta maps.
- Rank ROIs by response delta magnitude.
- Cluster condition-level response deltas to inspect separability.

## 5. Results

- Report global response delta magnitude per condition.
- Report top shifted ROIs per condition.
- Show condition clustering or low-dimensional projections.
- Compare steered conditions against prompt-only baselines where applicable.
- Use cautious language around model-predicted effects.

## 6. Discussion

- Interpret whether steering conditions produce structured predicted cortical-response changes.
- Discuss what observed response deltas may imply about the model's learned representation.
- Identify where results are consistent, unstable, or difficult to interpret.

## 7. Limitations

- Predicted cortical responses are model outputs, not direct human measurements.
- The study does not infer real emotions, thoughts, or psychological states.
- Results may be sensitive to stimulus choice, steering wording, model architecture, and ROI metadata.
- Empirical validation is required before broader scientific claims.

## 8. Ethics

- Explain prohibited uses such as diagnosis, surveillance, hiring, education scoring, law enforcement, and medical decision-making.
- Clarify upstream model and asset license responsibilities.
- Emphasize transparent reporting of steering definitions, limitations, and validation status.
