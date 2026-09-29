# Export the curated public breast cancer cohorts of case studies IV and V (MetaGxBreast).
#   Rscript data_prep/export_curated_cohorts.R <data_dir> [breast]
#
# For every dataset this writes <data_dir>/cohorts/<cancer>/<dataset>/clinical.csv
# (one row per sample: identifiers, endpoints, clinical covariates) and
# expression.csv.gz (samples x genes). Two tables describe the export:
# - <data_dir>/cohorts/index.csv, one line per dataset with sample counts, platform and, per
#   endpoint, the number of samples with follow-up and events.
# - <data_dir>/cohorts/manifest.csv, one line per file written, with its SHA-256 and size, the
#   package and version it came from, the ExperimentHub record and snapshot date where there is
#   one, the R and Bioconductor versions and the time of the export.
# Both are rewritten after each dataset: the lines of the cancers exported in this run are
# replaced by this run's datasets, the other cancers' lines are kept.
suppressPackageStartupMessages({
  library(Biobase)
  library(data.table)
})
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: Rscript export_curated_cohorts.R <data_dir> [breast]")
data_dir <- args[[1]]
cancers <- if (length(args) > 1) args[-1] else "breast"
out_root <- file.path(data_dir, "cohorts")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
index_path <- file.path(out_root, "index.csv")
manifest_path <- file.path(out_root, "manifest.csv")
exported_at <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
bioconductor <- tryCatch(as.character(BiocManager::version()), error = function(e) NA_character_)
index_rows <- list()
manifest_rows <- list()

as_expression_set <- function(object) {
  if (is(object, "ExpressionSet")) return(object)
  if (is(object, "SummarizedExperiment")) {
    return(ExpressionSet(
      assayData = SummarizedExperiment::assay(object),
      phenoData = AnnotatedDataFrame(as.data.frame(SummarizedExperiment::colData(object)))
    ))
  }
  stop("Unsupported object class: ", class(object)[1])
}

endpoint_counts <- function(pheno, time_column, status_column, event_value) {
  if (!all(c(time_column, status_column) %in% names(pheno))) return(c(n = 0L, events = 0L))
  time <- suppressWarnings(as.numeric(pheno[[time_column]]))
  status <- as.character(pheno[[status_column]])
  usable <- !is.na(time) & time >= 0 & !is.na(status)
  c(n = sum(usable), events = sum(usable & status %in% event_value))
}

package_version_of <- function(package) {
  if (requireNamespace(package, quietly = TRUE)) as.character(packageVersion(package)) else NA_character_
}

# Where a dataset came from: the package and, for ExperimentHub, the record and snapshot.
provenance <- function(package, record_id = NA_character_, record_title = NA_character_, hub_snapshot = NA_character_) {
  list(package = package, package_version = package_version_of(package), record_id = record_id,
       record_title = record_title, hub_snapshot = hub_snapshot)
}

record_file <- function(cancer, dataset, path, source) {
  manifest_rows[[length(manifest_rows) + 1]] <<- data.table(
    cancer = cancer, dataset = dataset, file = substring(path, nchar(out_root) + 2),
    sha256 = unname(tools::sha256sum(path)), bytes = file.size(path),
    package = source$package, package_version = source$package_version, record_id = source$record_id,
    record_title = source$record_title, hub_snapshot = source$hub_snapshot,
    r_version = R.version.string, bioconductor = bioconductor, exported_at = exported_at
  )
}

# Replace the lines of the cancers exported in this run, keep the other cancers' lines.
update_table <- function(path, rows) {
  if (!length(rows)) return(invisible(NULL))
  table <- rbindlist(rows, fill = TRUE)
  if (file.exists(path)) {
    # Read as text, so no column takes a type (a date, say) this run's values do not fit.
    previous <- fread(path, colClasses = "character")
    previous <- previous[!(previous$cancer %in% cancers)]
    if (nrow(previous)) table <- rbindlist(list(previous, table[, lapply(.SD, as.character)]), fill = TRUE)
  }
  fwrite(table, path)
}

write_dataset <- function(eset, cancer, name, platform, source) {
  eset <- as_expression_set(eset)
  pheno <- pData(eset)
  pheno$sample_id <- sampleNames(eset)
  folder <- file.path(out_root, cancer, name)
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  clinical_path <- file.path(folder, "clinical.csv")
  expression_path <- file.path(folder, "expression.csv.gz")
  fwrite(pheno, clinical_path)
  expression <- t(exprs(eset))
  gene_names <- if ("gene" %in% names(fData(eset))) as.character(fData(eset)$gene) else colnames(expression)
  # Symbols are written as the dataset gives them: GSE58644 pads its symbols with spaces (" RNF207 "), so a gene's
  # further probes are named " RNF207 .1", ...; the Python loader (common.probe_gene) reads them.
  colnames(expression) <- make.unique(ifelse(is.na(gene_names) | gene_names == "", colnames(expression), gene_names))
  fwrite(data.table(sample_id = rownames(expression), expression), expression_path)
  os <- endpoint_counts(pheno, "days_to_death", "vital_status", c("deceased", "dead", "1"))
  rfs <- endpoint_counts(pheno, "days_to_tumor_recurrence", "recurrence_status", c("recurrence", "1"))
  dmfs <- endpoint_counts(pheno, "dmfs_days", "dmfs_status", c("deceased_or_recurrence", "recurrence", "1"))
  index_rows[[length(index_rows) + 1]] <<- data.table(
    cancer = cancer, dataset = name, platform = platform, n_samples = ncol(eset), n_genes = nrow(eset),
    os_n = os[["n"]], os_events = os[["events"]],
    rfs_n = rfs[["n"]], rfs_events = rfs[["events"]],
    dmfs_n = dmfs[["n"]], dmfs_events = dmfs[["events"]]
  )
  record_file(cancer, name, clinical_path, source)
  record_file(cancer, name, expression_path, source)
  update_table(index_path, index_rows)
  update_table(manifest_path, manifest_rows)
  cat(cancer, name, ncol(eset), "samples,", nrow(eset), "genes\n")
}

platform_of <- function(eset) {
  eset <- as_expression_set(eset)
  value <- tryCatch(annotation(eset), error = function(e) "")
  if (length(value) == 0 || identical(value, "")) "unknown" else paste(value, collapse = ";")
}

if ("breast" %in% cancers) {
  # loadBreastEsets() returned no datasets under MetaGxBreast 1.30 (it exits cleanly with an
  # empty list), so each ExpressionSet is fetched from ExperimentHub by its title instead.
  # METABRIC and TCGA are included; cross-dataset duplicates are listed in each sample's
  # "duplicates" column and in breast_duplicates.csv, and screened in the analysis.
  hub <- ExperimentHub::ExperimentHub(ask = FALSE)
  snapshot <- as.character(AnnotationHub::snapshotDate(hub))
  breast <- AnnotationHub::query(hub, c("MetaGxBreast", "ExpressionSet"))
  for (id in names(breast)) {
    eset <- hub[[id]]
    title <- breast[id]$title
    write_dataset(eset, "breast", title, platform_of(eset), provenance("MetaGxBreast", id, title, snapshot))
    rm(eset)
    gc()
  }
  load(system.file("extdata", "duplicates.rda", package = "MetaGxBreast"))
  duplicates_path <- file.path(out_root, "breast_duplicates.csv")
  fwrite(data.table(sample = names(duplicates), duplicates = vapply(duplicates, paste, "", collapse = ";")), duplicates_path)
  record_file("breast", "duplicates", duplicates_path, provenance("MetaGxBreast", record_title = "extdata/duplicates.rda"))
  update_table(manifest_path, manifest_rows)
}
cat("MetaGxBreast", package_version_of("MetaGxBreast"), "\n")
