# Limitations and Ethics

ASNE is a computational research framework. It should be used to study predicted cortical responses from models under defined steering conditions, not to infer true emotions, thoughts, intentions, or mental states.

## Core Limitations

- Predicted cortical responses are model outputs, not direct human neural measurements.
- Response deltas reflect changes under computational conditions, not detected emotions or verified psychological states.
- Steering conditions are experiment labels and controls, not clinical, diagnostic, or personal attributes.
- Results may depend on the upstream model, stimulus choice, preprocessing, ROI definitions, and steering mechanism.
- Empirical neuroscience claims require validation against measured human neural data.
- TODO: Add TRIBE v2-specific limitations after reviewing its training data, intended use, output structure, and license.

## Prohibited or Inappropriate Uses

ASNE should not be used for:

- Psychological diagnosis.
- Emotional or cognitive surveillance.
- Manipulation or persuasion targeting.
- Hiring, promotion, or workplace evaluation.
- Education scoring or student profiling.
- Law enforcement, security screening, or legal decision-making.
- Medical triage, treatment selection, or other medical decision-making.

## Simulation Versus Empirical Neuroscience

ASNE produces computational simulations of predicted cortical-response patterns. These predictions may be useful for model analysis, hypothesis generation, and methodological research, but they are not substitutes for empirical neuroscience experiments.

Any scientific claim should clearly distinguish:

- Model-predicted cortical responses from measured neural responses.
- Steering conditions from real psychological states.
- Response deltas from detected emotions or intentions.
- Exploratory findings from validated empirical results.

## Citation and Validation Expectations

Researchers should cite upstream models, datasets, steering methods, neural encoding literature, and any empirical validation data used in analysis. Claims should be constrained to the tested model and protocol unless independent validation supports broader interpretation.

Before publication or deployment, ASNE experiments should document:

- Upstream model version and license.
- Stimulus source and license.
- Steering condition definitions.
- Baseline and steered inference settings.
- Evaluation metrics.
- Known limitations and failure cases.
