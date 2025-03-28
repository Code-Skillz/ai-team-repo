# Alternative CodeSearchNet Dataset

This directory contains an alternative dataset for debugging insights, created to replace the CodeSearchNet dataset which had Windows path compatibility issues.

## Contents

- 10 debugging examples across 7 programming languages
- Each example includes code with an error, error message, debugging hint, and fixed code
- Data is formatted for T5 fine-tuning

## Files

- `debugging_examples_raw.json`: Raw debugging examples
- `debugging_dataset_t5.json`: Full dataset formatted for T5 fine-tuning
- `train.json`: Training split (80%)
- `val.json`: Validation split (10%)
- `test.json`: Test split (10%)
