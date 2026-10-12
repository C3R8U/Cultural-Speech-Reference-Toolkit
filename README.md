# Cultural Speech Semantic Reference Toolkit

A newly developed reference implementation of the method described in “End-to-End Speech-Semantic Model for Immersive Cultural Content Generation.” It includes model and training code, CTC decoding and semantic reranking, statistical analysis, dynamic INT8 quantization, data checks and collection scaffolds.

**Availability:** This release contains no original experimental source code, trained weights, real CSSP/Field recordings, human transcripts, prediction logs or participant ratings. Both supplied cultural archives are uncollected scaffolds. This toolkit enables method inspection and future experiments with completed data; it does not reproduce or verify the manuscript's numerical findings. No public repository URL exists in this archive before upload. See docs/AVAILABILITY_AND_REVIEWER_RESPONSE.md for truthful manuscript and reviewer-response wording before and after publication.

## Install and test

Use Python 3.10–3.12 (local verification: Python 3.12, Windows CPU). Create an isolated environment, activate it and install:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
python tools/check_data.py data/collection_scaffolds/CSSP data/collection_scaffolds/Field-Culture_Test --mode scaffold --out scaffold_report.json
```

Running the same check with --mode research is expected to exit 2 because recordings and annotations are missing. This is a successful rejection of unusable research input, not a complete-dataset validation result. Tests synthesize small tensors and a sine wave in temporary directories solely to check algorithms; they do not simulate study participants or report paper metrics.

## Execute with completed data

Read docs/DATA.md, prepare actual verified records and retain source licenses. Create train and validation manifests without speaker overlap. Supply real public corpus files separately; no downloads occur automatically.

```sh
python tools/build_vocab.py actual_speech.jsonl news.jsonl poetry.jsonl --out vocab.json
python -m culture_speech.pretrain_text --input news.jsonl --task semantic --vocab vocab.json --out runs/news.pt --seed 17
python -m culture_speech.pretrain_text --input poetry.jsonl --task generation --checkpoint runs/news.pt --out runs/text.pt --seed 17
python -m culture_speech.train --manifest actual_speech.jsonl --root actual_audio_root --checkpoint runs/text.pt --out runs/seed17 --seed 17
python -m culture_speech.evaluate --checkpoint runs/seed17/best.pt --manifest actual_speech.jsonl --root actual_audio_root --out runs/predictions17.csv --unit character --alpha 0.1
python -m culture_speech.quantize --checkpoint runs/seed17/best.pt --out runs/model_int8.pt
python -m culture_speech.evaluate --checkpoint runs/seed17/best.pt --manifest actual_speech.jsonl --root actual_audio_root --out runs/predictions17_int8.csv --int8 --unit character --alpha 0.1
```

Replace example paths with actual files. Pretraining is optional; omit --checkpoint for random initialization. Keep the same train-only vocabulary throughout the stages. The auxiliary news head is saved separately; it is not the cultural intent head. Supply the exact original curriculum if available; no schedule is invented here as the recovered experiment. Repeat independently with seeds 17, 29, 42, 73 and 101, and concatenate prediction CSVs with a single header. Tune alpha on validation only. For controlled bridges, use train --bridge soft or --bridge hard, with matched data/seeds; evaluate hard/soft controls with --no-rerank. A standalone INT8 file loads through culture_speech.quantize.load_int8; evaluation --int8 regenerates the same quantization from FP32 using the local backend.

## Statistical procedures

```sh
python -m culture_speech.statistics errors paired_predictions.csv --baseline reference_dcsb --iterations 10000 --out wer_statistics.json
python -m culture_speech.statistics ratings ratings.csv --out ratings_statistics.json
Rscript statistics/intent_glmm.R intent_predictions.csv runs/intent_glmm
```

Error CSV columns: model,seed,sample_id,errors,reference_units,unit,reference. Use one consistent tokenizer/unit and a complete paired grid. --baseline defines the Holm comparison family. For ratings, columns are unit_id,model,score, one row per participant/request and model, scores 1–5. Prepare one file per dimension, or preregister a family across dimensions. Average evaluators before request-level analysis. R requires lme4 and emmeans; the GLMM script emits diagnostics and rejects singular/nonconvergent fits. Do not infer p-values from summary means or fabricate missing per-sample results.

## Data and device checks

```sh
python tools/resample.py actual_raw.wav actual_16k.wav
python tools/deduplicate.py --training-texts poetry_training_targets.txt --paired-manifest actual_speech.jsonl --out leakage.json
python tools/benchmark.py --checkpoint runs/seed17/best.pt --wav-list actual_100_three_second_wavs.txt --int8 --threads 4 --out benchmark.json
```

Device benchmarking is local and excludes generation and MFCC extraction; see docs/METHOD_MAPPING.md for its exact scope. Test on the real Raspberry Pi separately. No original power/latency values are inserted. Collection consent and identity checks must be completed using actual recruitment records.

## Publish

Upload this directory's contents to a new public GitHub repository; retain availability statements and source provenance. Run the included GitHub Actions workflow and inspect its logs. Tag a release and record the commit hash. Confirm access while signed out, then replace the editing tokens in docs/AVAILABILITY_AND_REVIEWER_RESPONSE.md with the actual link/release/commit in both manuscript and response letter. An upload of this toolkit does not establish that original datasets or experimental results exist. Resolve the manuscript's completed-data claims against genuine original evidence before submission.

Code reuse terms are in CODE_LICENSE.txt. Source scaffold files remain separately attributed under data/collection_scaffolds and have no supplied redistribution license; obtain the rights holder's permission before public redistribution. To publish code without redistributing them, omit data/collection_scaffolds, its source-data audit reports and scaffold tests; document this change and rerun the remaining checks. Public corpus licenses remain with their providers. Review the local verification report in verification/ and docs/METHOD_MAPPING.md for untested components and reproducibility limits.
