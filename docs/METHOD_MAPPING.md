# Method mapping and limitations

| Manuscript component | Package implementation |
|---|---|
| MFCC 13 + first/second differences | data.py, 39 features, 25 ms window/10 ms hop |
| Three-layer BiLSTM and CTC | model.py, packed variable-length inputs |
| Blank-aware DCSB | model.py, nonblank renormalization, confidence-gated input and pooling |
| Semantic LSTM and three intents | model.py, two layers and three-way head |
| Two-layer generation decoder | model.py, teacher forcing and autoregressive beam search |
| Joint objective .60/.25/.15 | model.py joint_loss |
| CTC candidates and semantic consistency | decoding.py, beam 10, top five reranking |
| Training seeds and early stopping | train.py, seed chosen explicitly; run five commands |
| General text auxiliary tasks | pretrain_text.py with independent temporary news-class head |
| Seed confidence intervals | statistics.py Student t with n-1 degrees of freedom |
| Paired two-level WER statistics | statistics.py null-centred crossed seed/sentence bootstrap and Holm |
| Crossed logistic intent model | statistics/intent_glmm.R, lme4 LRT and emmeans contrasts |
| Friedman, Kendall W, Wilcoxon | statistics.py, complete paired rating matrices |
| Dynamic INT8 | quantize.py, CPU LSTM/Linear quantization |
| Data checking and reference leakage | tools/check_data.py and tools/deduplicate.py |

New implementation choices include n_fft=512, 40 Mel filters, Hann window, no centering, width-9 deltas, confidence-weighted pooling, decoder initial hidden state from pooled semantics, raw CTC log probability plus alpha*cosine consistency, no length penalty and reference metric normalization. The supplied extracted document does not expose all displayed mathematical equations or original code. These choices implement a coherent method but cannot be certified as an exact transcription of every formula. Select alpha on validation only; default .1 is a reference choice, not a recovered paper setting.

All comparison backbones, external checkpoints and pretrained BERT pipelines are outside this release. The package does not claim to reproduce their tables. The proposed text pretraining workflow supplies a news auxiliary head and poetry continuation pretraining; the original curriculum and datasets must be recovered to match reported results. Random initialization and toy tests cannot substitute for real corpus evaluation. GPU deterministic execution may require CUBLAS_WORKSPACE_CONFIG=:4096:8; unsupported deterministic operators should raise errors rather than silently changing the protocol.

INT8 code is tested with PyTorch 2.5.1 CPU rather than the paper's 2.1/CUDA12.1 environment. Embeddings, MFCC, softmax, beam search and unsupported operations remain floating point. No offline calibration is required for dynamic quantization. CPU state files are version/backend-specific. x86 checks are not evidence for Raspberry Pi latency, RAM or power. benchmark.py measures feature-input through semantic reranking; generation and MFCC timing are excluded and the output says so. It cannot directly reproduce a broader end-to-end latency definition. Sampled RSS is a process measurement, not exact device peak memory; power requires an external meter. FLOPs and cultural fact accuracy are not estimated by invented constants.

Bootstrap input requires one row per model/seed/sentence, the same references across systems, positive reference lengths and complete grids. Each bootstrap draw uses paired common seed and sentence draws and computes mean corpus error-rate differences across drawn seeds. Percentile CIs and a finite-sample-corrected null-centred two-sided p-value are reported. The precise original p-value construction was not supplied, so this explicitly stated choice is a reference procedure. Sentence-level sampling follows the paper description; speaker clustering is a sensitivity analysis still needed for speaker-dependent utterances. Five seeds provide limited precision.

Rating files represent one participant (subjective ratings) or one request (generation ratings) per unit and model. Average the three evaluators' scores for each request before input, without treating raters as independent requests. Analyze each rating dimension separately and declare multiplicity families. Wilcoxon uses asymptotic tie handling. GLMM fails on incomplete data, singularity or convergence problems; five seed levels may make random-effect estimates unstable. R code was executed locally on an explicitly artificial fixture; this checks execution, not manuscript findings. Record R/package versions and validate with original outcomes before use.

Official implementation references:

- https://docs.pytorch.org/docs/2.5/generated/torch.nn.CTCLoss.html
- https://docs.pytorch.org/docs/2.5/generated/torch.ao.quantization.quantize_dynamic.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- https://stat.ethz.ch/R-manual/R-devel/library/stats/html/friedman.test.html
- https://cran.r-project.org/package=lme4
- https://cran.r-project.org/package=emmeans
