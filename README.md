# Appendix for “中国文化短视频多语传播机制研究”

This repository contains public, reproducible appendix materials for the paper:

> 中国文化短视频多语传播机制研究

The appendix is organized for review and reproduction. It includes the questionnaire, anonymized/derived survey data, analysis scripts, statistical tables, figures, and anonymized qualitative materials.

## Contents

- `questionnaire/`
  - `second_round_questionnaire.pdf`: the second-round questionnaire used for the formal survey.

- `data/`
  - `second_round_cleaned_with_scores.csv` and `.xlsx`: anonymized second-round survey data with derived dimension scores.
  - `dimension_scores_long.csv`: long-format dimension-score table for repeated-measures analysis and plotting.
  - `data_dictionary.csv`: variable descriptions and scale direction.
  - `data_validation_checks.csv`: validation checks for missing values and expected scale ranges.

- `analysis/`
  - `analysis_summary.md`: concise quantitative results summary.
  - `scripts/run_quant_analysis.py`: reproducible main quantitative-analysis script.
  - `scripts/run_efa_checks.py`: supplementary KMO, Bartlett, parallel-analysis, EFA, and adjusted-correlation script.
  - `tables/`: descriptive statistics, reliability, dimension-difference tests, group tests, correlations, regression, EFA, and consolidated workbook outputs.
  - `figures/`: exported figures used or available for the report.

- `qualitative/`
  - `anonymized_quotes.md`: anonymized pilot-study quotes.
  - `theme_summary.md`: pilot-study theme summary.
  - `coding_table.xlsx`: anonymized coding table.

## Reproducibility

The main quantitative workflow can be reproduced from the project root with:

```bash
python3 analysis/scripts/run_quant_analysis.py
python3 analysis/scripts/run_efa_checks.py
```

The scripts are included for transparency. They read survey workbooks in the original local project, generate derived clean data, and write statistical tables and figures. The GitHub appendix contains the derived data and outputs needed to inspect the reported results.

## Public-Release Boundary

The appendix intentionally excludes files that are not appropriate for public redistribution:

- first-round raw questionnaire workbook containing direct identifiers such as names and contact information;
- original raw response workbooks with possible private metadata;
- literature records, bibliographic files, and downloaded full-text literature PDFs.

Instead, this public appendix provides the questionnaire, anonymized/derived data, reproducible scripts, statistical outputs, figures, and anonymized qualitative materials.
