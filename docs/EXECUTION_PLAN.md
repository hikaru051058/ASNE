# Execution Plan

## Phase 1: Repo Scaffolding

- Create project documentation, ethics notes, and example configuration files.
- Define terminology around steering conditions, predicted cortical responses, and response deltas.
- Establish initial file layout for future experiments, configs, outputs, and reports.
- TODO: Decide final package structure after confirming the TRIBE v2 runtime requirements.

## Phase 2: TRIBE v2 Baseline Runner

- Load TRIBE v2 or a similar neural-response foundation model as a frozen upstream dependency.
- Run baseline inference with no steering condition or with the neutral steering condition.
- Save predicted cortical-response outputs in a reproducible format.
- Capture model version, checkpoint path, stimulus metadata, and inference parameters.
- TODO: Implement once the actual TRIBE v2 API, input schema, and output schema are available.

## Phase 3: Steering Condition Config

- Load steering condition definitions from YAML.
- Validate required fields such as `id`, `name`, `description`, `prompt_prefix`, `intended_analysis`, and `safety_note`.
- Support neutral and non-neutral steering conditions under a shared schema.
- Add metadata for future steering methods such as soft prompts, activation vectors, representation steering layers, or adapters.

## Phase 4: Steered Inference Runner

- Apply a selected steering condition to the same stimulus used for baseline inference.
- Run frozen-model inference under the steering condition.
- Save outputs with condition identifiers and reproducibility metadata.
- Ensure baseline and steered runs are comparable across stimulus, model version, and preprocessing.
- TODO: Define the exact steering hook after reviewing TRIBE v2 internals.

## Phase 5: Delta Analysis

- Compute response deltas between baseline and steered predicted cortical responses.
- Summarize global delta magnitude and ROI-level response shifts.
- Rank ROIs by response delta magnitude or condition sensitivity.
- Compare deltas across steering conditions and stimuli.
- Include controls for prompt-only effects where possible.

## Phase 6: Visualization and Report Generation

- Generate response-delta plots, ROI rankings, and condition comparison figures.
- Produce compact experiment reports with stimulus metadata, steering definitions, metrics, and limitations.
- Export figures and tables for paper-style review.
- TODO: Select visualization formats after confirming response tensor shape and ROI metadata.

## Phase 7: Preliminary Paper-Style Results

- Run a first controlled experiment with one short video and six steering conditions.
- Compare baseline versus steered predicted cortical responses.
- Produce delta maps, ROI rankings, and condition clustering.
- Write a preliminary results section with cautious interpretation.
- Identify failure modes, confounds, and validation requirements before making scientific claims.
