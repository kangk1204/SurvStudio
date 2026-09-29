# Data snapshot

Three small files the paper's data chain needs that their sources replace over time. No other data are
redistributed: `prepare_data.sh` downloads the rest from the public sources (see `../README.md`). `common.check_snapshot`
(`scripts/common.py`), run by `00_self_check.py` and `prepare_data.sh`, checks each file against the SHA-256 below.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Homo_sapiens.gene_info.gz` | 5,190,133 | `4a333625190f594abc21fd7564c7a0ee991a0331c639c6d3b7bb814b944e75b7` |
| `cbioportal_patients.csv` | 410,491 | `f09389e31e88b658db463eb33e716fa7c0c905c4f2527c76cc87ccb5e02c0bee` |
| `sample_decisions.csv` | 12,037 | `529aa8bc7ab24c5065ce32e1fbddc1bd4c0c470c764a45150c0a4ac0cdc2e8a6` |

## Homo_sapiens.gene_info.gz

NCBI Gene's table of human genes (Entrez Gene ID, official symbol, synonyms), downloaded on 2026-09-26 from
<https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz>, unchanged. NCBI replaces this
file daily; `data_prep/harmonize_expression.py` maps the GEO probes to gene symbols with this copy, so the gene-level
expression files match `luad_data_manifest.csv`, and it stops on any other file.

Licence: NCBI Gene data are in the public domain; NCBI places no restrictions on their use or distribution
(<https://www.ncbi.nlm.nih.gov/home/about/policies/>). Cite NCBI Gene: Brown GR et al., Gene: a gene-centered
information resource at NCBI, Nucleic Acids Research 43:D36-D42 (2015).

## cbioportal_patients.csv

The METABRIC patient table of cBioPortal's `brca_metabric` study (2,509 patients, one row each; overall survival
`OS_MONTHS`/`OS_STATUS`, relapse-free survival `RFS_MONTHS`/`RFS_STATUS`, the site `COHORT` and the other patient
attributes), downloaded on 2026-09-28 from
<https://www.cbioportal.org/api/studies/brca_metabric/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY> and
written with one row per patient by `common.metabric_cbioportal`. Case studies IV and V take METABRIC's endpoints and
sites from it. `common.metabric_cbioportal` downloads the table on first use; when the download differs from this copy
(cBioPortal updates the study from time to time) or fails, it uses this copy.

Licence: this table is an extract of the cBioPortal database and is made available under the Open Database License
(ODbL) v1.0, <https://opendatacommons.org/licenses/odbl/1-0/>; any rights in individual contents of the database are
licensed under the Database Contents License, <https://opendatacommons.org/licenses/dbcl/1-0/>. Attribution: the METABRIC
study (Curtis C et al., Nature 486:346-352, 2012; Pereira B et al., Nature Communications 7:11479, 2016), obtained from
cBioPortal for Cancer Genomics (Cerami E et al., Cancer Discovery 2:401-404, 2012; Gao J et al., Science Signaling
6:pl1, 2013; de Bruijn I et al., Cancer Research 83:3861-3867, 2023).

## sample_decisions.csv

The QC decisions the paper's harmonized LUAD table applies: 223 samples of the eight LUAD cohorts (TCGA-LUAD and the
seven GEO series) with a QC exclusion (`qc_exclusion`: technical duplicate, annotated sex contradicting expression,
PanCanAtlas quality annotations; 107 samples) or a sensitivity flag (`qc_sensitivity`), written by
`data_prep/qc_cohorts.py` on 2026-09-26. `prepare_data.sh` copies it to `cohorts/luad/qc/` before
`data_prep/harmonize_luad.py` runs, reruns the QC and reports whether the rerun decides the same.

Licence: derived from the public TCGA-LUAD and GEO data listed in `../README.md`; released with this repository under
its MIT License. The identifiers are TCGA sample barcodes and GEO sample accessions.
