# Availability and reviewer response

The archive is ready for upload but is not a public repository until the uploader publishes it. No URL or DOI has been invented. This package was newly implemented from the supplied manuscript; it is not recovered original experiment code. The two supplied data archives contain collection scaffolds, zero real recordings and zero human transcripts. Original checkpoints, prediction logs, participant responses and experiment provenance were not supplied. These limitations must remain visible after upload.

## Text accurate before public upload

Experimental section replacement:

“A reference software package has been prepared for the MFCC–BiLSTM–CTC model, differentiable speech–semantic bridge, joint training, semantic reranking, statistical testing, dynamic INT8 quantization and data validation. The package was implemented from the described methodology and is not the archived implementation used to obtain the reported results. It has not yet been deposited in a publicly accessible repository. The available CSSP and Field-Culture Test materials are collection and annotation scaffolds and contain no recorded speech or human transcriptions. These materials do not permit independent reproduction of the reported experimental results.”

Data Availability Statement replacement:

“The public source locations for AISHELL-1, Common Voice Chinese, THUCNews and Chinese Poetry are listed in the data documentation. The exact study subsets and preprocessing records have not been deposited. The available CSSP and Field-Culture Test materials comprise uncollected task and annotation scaffolds rather than completed participant datasets. Original recordings, trained checkpoints and participant-level experimental results are not included in the prepared reference package. The package has not yet been publicly deposited. The availability of original experimental materials must be confirmed before making any claim that they can be obtained upon request.”

Response to Reviewer 1 before upload:

“Thank you for identifying this inconsistency. We have removed the unsupported statement that a public repository was explicitly linked in the experimental section. A reference software package has been prepared, but no accessible repository link is currently available. We have revised both the experimental section and the Data Availability Statement to distinguish the reference implementation and uncollected data scaffolds from the original experimental materials. The package does not establish reproducibility of the reported numerical results. We will provide a public URL only after deposition and access verification.”

## Text to complete after upload and access verification

Replace REPOSITORY_URL with the actual public HTTPS URL, and RELEASE_TAG and COMMIT_SHA with the published release identifier and commit. If a DOI is created through an archival service, add the actual DOI. These are editing tokens, not links; do not submit this template unchanged.

Experimental section:

“The reference implementation and data-structure validation tools are publicly available at REPOSITORY_URL (release RELEASE_TAG, commit COMMIT_SHA). The repository contains the DCSB model, joint training and decoding code, statistical-testing procedures, dynamic INT8 quantization tools, collection scaffolds and usage instructions. It is a newly developed reference implementation of the described methodology. Original trained checkpoints and per-sample experiment results are not included, and the included CSSP and Field-Culture Test scaffolds contain no real recordings or human transcriptions. Consequently, the repository supports method inspection and future execution with completed data, but does not independently reproduce the reported numerical results.”

Data Availability Statement:

“Reference code, collection scaffolds and documentation are available at REPOSITORY_URL (release RELEASE_TAG, commit COMMIT_SHA). AISHELL-1, Common Voice, THUCNews and Chinese Poetry are distributed by their respective providers; their source links and reuse conditions are documented in the repository. Exact study subsets, original CSSP and Field-Culture Test recordings, original trained weights and participant-level results are not included. The supplied cultural corpus materials are uncollected scaffolds. Any future release of participant recordings will require completion of collection and confirmation of consent and appropriate access conditions.”

Response to Reviewer 1 after actual public deposition:

“Thank you for pointing out the inconsistency. We have now provided the accessible repository link REPOSITORY_URL in the experimental section and Data Availability Statement, and identified release RELEASE_TAG and commit COMMIT_SHA. We have also corrected the scope of the availability claim: the repository contains a newly developed reference implementation, statistical and quantization tools, data checks and uncollected cultural corpus scaffolds. It does not contain original recordings, trained checkpoints or per-sample results and therefore does not establish independent reproduction of the reported numerical results. The manuscript and response letter now describe these limitations consistently.”

## Manuscript evidence that requires author resolution

The manuscript describes CSSP and Field-Culture Test as completed collections and reports corresponding model results. The provided archives explicitly label themselves as uncollected scaffolds. A repository upload alone cannot resolve this discrepancy. Locate and validate genuine recorded corpora, consent and annotation records, weights and experiment logs if they exist; otherwise revise or remove unsupported sample-collection and empirical-result claims in all affected sections, tables and figures. This package cannot certify the paper's reported findings.

Before submission, verify the actual public URL in a signed-out browser, upload a versioned release, retain the README availability statement, ensure the methods section and Data Availability Statement use the same URL and scope, and remove unfilled editing tokens. No response text should imply public deposition before it has occurred.
