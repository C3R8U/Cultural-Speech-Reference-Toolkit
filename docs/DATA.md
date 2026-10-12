# Data sources and local format

The supplied CSSP.zip and Field-Culture_Test.zip are included without changing their scientific status in data/collection_scaffolds. Their manifests say collection_ready_scaffold, with real_audio_count=0 and human_transcript_count=0. Task prompts are synthetic collection materials; labels are planned categories, not observations. Reserved speaker IDs do not prove real-person independence. The source archives' original README and manifests are preserved.

Public resources (download separately and retain the provider's license and release metadata):

| Resource | Provider | Paper use |
|---|---|---|
| AISHELL-1 | https://www.openslr.org/33/ | Official ASR splits, approximately 178 h |
| Common Voice Chinese v16 | https://commonvoice.mozilla.org/en/datasets | Paper reports 44 h retained, split 36/4/4 h; exact subset IDs unavailable |
| THUCNews | https://github.com/thunlp/THUCTC | Paper reports 90,000 articles, split 70,000/10,000/10,000 |
| Chinese Poetry | https://github.com/chinese-poetry/chinese-poetry | Paper reports 60,000 sequences, split 50,000/5,000/5,000 |

KeSpeech (https://github.com/tzyll/KeSpeech) is a distinct dataset, not a Common Voice source, and is not used by this implementation. Provider pages alone cannot identify the exact paper subsets; do not label newly sampled subsets as the original experiment data. No public corpora or pretrained third-party weights are redistributed here.

## Completed speech record contract

Supply UTF-8 JSONL records with sample_id, speaker_id, split (train/validation/test), model_audio_path (relative to a root), transcript (actual human-verified speech), cultural_intent_label (文化咨询/情感表达/创意请求, or omitted for ASR-only data), and status='collected'. Creative requests additionally require generation_subtype (古诗续写/文化故事生成), top-level reference_text and reference_verified=true. Original nested collection references are prompts or provisional reference materials; they are not automatically converted into verified training targets.

Model audio must be uncompressed PCM WAV, mono, 16 kHz and 16 bits. Raw recordings are mono 48 kHz/24-bit; tools/resample.py uses a polyphase antialiasing filter, not a header change. Paper-duration checks use 1.5–12 seconds. Record original annotation history, independent review and disagreements externally. Train/validation/test speakers must be disjoint. Confirm actual-person independence between CSSP and Field data using controlled recruitment records; never upload identity mappings.

check_data.py reads the original scaffold layout and validates counts, IDs, splits, categories, scenes and PCM headers. Its --mode scaffold only certifies structural checks. --mode research additionally requires actual WAVs, transcripts, labels, verified references, collected status, audio_present=true, and speaker metadata: adult_verified=True, consent_status=obtained, identity_dedup_status=verified. It detects duplicate model waveform hashes. It does not certify consent authenticity, expert factual verification, clipping quality or near-duplicate audio. Finish these manual audits before research use. Field data remain test-only.

## Text pretraining contract

Semantic JSONL: {"split":"train", "text":"actual news text", "label":"news category"}. Validation is mandatory; label classes must appear in training. Generation JSONL: {"split":"train", "prompt":"poetry prefix", "target":"verified continuation"}. Use independent works and deduplicate across splits and against CSSP evaluation references. Exact normalized deduplication is provided; add source-work and near-duplicate audits. Neither tool invents the original article/poem sampling rules.

## Metric convention

Default evaluation uses normalized characters and labels the resulting rate as character error rate (CER). --unit word uses jieba segmentation and labels it word error rate (WER). The paper reports WER without sufficient tokenization detail; these defaults cannot be asserted to reproduce it. NFKC, whitespace and punctuation removal are explicit; numerical-expression normalization was not recoverable. Chinese BLEU-2 and ROUGE-L use character tokens; smoothing and averaging are documented in code. BERTScore is not included and must not be claimed as reproduced. No dummy WAV files, participant profiles or paper-result CSVs are supplied.
