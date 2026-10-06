# ASR Robustness Evaluation Under Controlled Acoustic Noise

## Objective

Measure how controlled synthetic acoustic noise affects the transcription
accuracy of a pretrained Whisper ASR model on a small subset of LibriSpeech
test-clean.

## Planned experiment

- **Dataset:** LibriSpeech test-clean (20 utterances for smoke testing; size configurable later).
- **Model:** pretrained `openai/whisper-tiny.en`.
- **Conditions:** clean audio, and synthetic noise at 20 dB, 10 dB and 5 dB SNR.
- **Metrics:** Word Error Rate (WER) and Character Error Rate (CER).

## Current status

The evaluation pipeline is implemented (`scripts/run_experiment.py`) and a
20-utterance pilot has been run on clean audio and synthetic Gaussian noise at
20, 10 and 5 dB SNR. Results are summarized in `results/summary.md` (see
"Analyzing results" below). This is a small single-seed pilot without confidence
intervals, so it does not support claims about real-world acoustic conditions.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Analyzing results

After `scripts/run_experiment.py` has written `results/robustness_results.json`, summarize it
offline (standard library only; no model, dataset or network access):

```bash
# Windows
python scripts\analyze_results.py --input results\robustness_results.json --output-dir results

# macOS / Linux
python scripts/analyze_results.py --input results/robustness_results.json --output-dir results
```

This writes `summary.md`, `wer_cer_by_snr.csv` and `wer_cer_by_snr.svg` into the output
directory. Corpus-level WER/CER are taken from the JSON; the report is validated first (same
utterance IDs and order in every condition, requested and measured SNR for noisy conditions)
and the command fails with a clear message if it is malformed.
