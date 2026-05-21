# Steering Conditions

Steering conditions are experimental controls applied during model inference. They are not real emotions, psychological diagnoses, or claims about a person. ASNE uses them to study response deltas in predicted cortical-response space.

## Neutral

- Description: A reference condition intended to preserve a plain, minimally framed interpretation of the stimulus.
- Example prompt prefix: `Analyze the stimulus in a neutral descriptive context.`
- Intended interpretation: Baseline or control condition for comparison against other steering conditions.
- Warning: Do not claim this represents an emotionally neutral human state.

## Threat

- Description: A condition that frames the stimulus in terms of possible risk, danger, or uncertainty.
- Example prompt prefix: `Analyze the stimulus while attending to possible threat, risk, or signs of danger.`
- Intended interpretation: Tests whether risk-oriented framing changes predicted cortical-response patterns.
- Warning: Do not claim the model detected fear, anxiety, danger perception, or a real threat response.

## Nostalgia

- Description: A condition that frames the stimulus through memory, familiarity, and past-oriented associations.
- Example prompt prefix: `Analyze the stimulus while emphasizing memory, familiarity, and past-oriented associations.`
- Intended interpretation: Tests whether memory-oriented framing changes predicted cortical-response patterns.
- Warning: Do not claim the model detected nostalgia or autobiographical memory.

## Social Judgment

- Description: A condition that frames the stimulus around social evaluation, reputation, group norms, or interpersonal interpretation.
- Example prompt prefix: `Analyze the stimulus while attending to social evaluation, reputation, and interpersonal judgment.`
- Intended interpretation: Tests whether socially framed context shifts predicted cortical responses.
- Warning: Do not claim the model inferred social beliefs, moral judgment, or social anxiety.

## Curiosity

- Description: A condition that frames the stimulus around novelty, exploration, uncertainty, and information seeking.
- Example prompt prefix: `Analyze the stimulus while emphasizing novelty, open questions, and information seeking.`
- Intended interpretation: Tests whether exploration-oriented framing produces distinguishable response deltas.
- Warning: Do not claim the model detected real curiosity or motivation.

## Cognitive Load

- Description: A condition that frames the stimulus as requiring focused attention, working memory, or complex reasoning.
- Example prompt prefix: `Analyze the stimulus while emphasizing focused attention, working memory, and task difficulty.`
- Intended interpretation: Tests whether task-demand framing changes predicted cortical-response patterns.
- Warning: Do not claim the model measured actual cognitive load or mental effort.
