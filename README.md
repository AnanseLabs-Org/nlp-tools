# asr-recipe

`asr-recipe` is a packaged Python CLI for inspecting ASR dataset schemas, running non-audio analyses, and exporting a recipe manifest for downstream Whisper notebook work.

## Quick Start

Install from a cloned repo:

```bash
pip install -e .
```

Install with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Install directly from GitHub:

```bash
pip install "git+https://github.com/fiifinketia/nlp-tools.git"
```

Install in Colab:

```python
!pip install "git+https://github.com/fiifinketia/nlp-tools.git"
```

If your package lives in a branch other than `main`:

```python
!pip install "git+https://github.com/fiifinketia/nlp-tools.git@your-branch-name"
```

Inspect one dataset in sample mode:

```bash
asr-recipe inspect --dataset waxal_aka_asr --sample-size 1000
```

Run a non-audio analysis:

```bash
asr-recipe analyze text-frequency --dataset waxal_aka_asr --split train --sample-size 5000
```

Export the handoff recipe manifest:

```bash
asr-recipe export-recipe \
  --dataset waxal_aka_asr \
  --dataset ghana_english_2700hrs \
  --analysis schema \
  --analysis text-length \
  --out output/recipe.json
```

The v1 CLI does not decode audio and does not export a processed dataset. Its main artifact is the JSON recipe manifest.

If `hf://` reads fail with an `ImportError` mentioning `fsspec`, reinstall after the current dependency set is updated:

```bash
uv pip install -e .
```

## Colab Flow

Install:

```python
!pip install "git+https://github.com/fiifinketia/nlp-tools.git"
```

Inspect a dataset:

```python
!asr-recipe inspect --dataset waxal_aka_asr --split train --sample-size 1000
```

Run text-frequency analysis:

```python
!asr-recipe analyze text-frequency --dataset ghana_english_2700hrs --split train --sample-size 20000 -q
```

Export a recipe manifest:

```python
!asr-recipe export-recipe \
  --dataset waxal_aka_asr \
  --dataset ghana_english_2700hrs \
  --dataset twi_trigrams_parallel \
  --analysis text-frequency \
  --analysis text-length \
  --split-policy train-val \
  --val-ratio 0.1 \
  --out /content/whisper_recipe.json
```

## Push Checklist

1. Initialize the repo if needed: `git init`
2. Commit the package files, not local artifacts
3. Push to GitHub
4. Test a fresh install with `pip install "git+https://github.com/<owner>/<repo>.git"`
5. Test the CLI in a clean environment or Colab
