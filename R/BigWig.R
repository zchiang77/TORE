cran_pkgs  <- c("data.table", "logger", "future", "furrr", "stringr")
bioc_pkgs  <- c("rtracklayer", "GenomicRanges")
cran_miss <- setdiff(cran_pkgs, rownames(installed.packages()))
if (length(cran_miss)) {
  install.packages(cran_miss,
    repos = "https://cloud.r-project.org", quiet = TRUE)
}
bioc_miss <- setdiff(bioc_pkgs, rownames(installed.packages()))
if (length(bioc_miss)) {
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cloud.r-project.org", quiet = TRUE)
  BiocManager::install(bioc_miss, ask = FALSE, update = FALSE, quiet = TRUE)
}
library(rtracklayer)
library(GenomicRanges)
library(data.table)
library(logger)
library(future)
library(furrr)
library(stringr)
resolution     <- 5000L
dir_base       <- "~/p2/data/"
save_base      <- file.path(dir_base, "ChIP.GRCh38",
                            paste0("resolution", resolution))
dir.create(save_base, showWarnings = FALSE, recursive = TRUE)
chr_autosomes  <- paste0("chr", 1:22)
chr_sex        <- c("chrX", "chrY")      
chr_list       <- c(chr_autosomes, chr_sex)
ts <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
fmt_duration <- function(secs) {
  if (is.na(secs) || is.infinite(secs) || secs < 0) return("--")
  h <- floor(secs / 3600)
  m <- floor((secs %% 3600) / 60)
  s <- round(secs %% 60, 1)
  if (h > 0) return(sprintf("%dh %dm %.1fs", h, m, s))
  if (m > 0) return(sprintf("%dm %.1fs", m, s))
  sprintf("%.1fs", s)
}
log_file <- file.path("~/p2", "BigWig_preprocess.log")
log_threshold(INFO)
log_appender(appender_tee(log_file))
log_info("============================================================")
log_info(" BigWig preprocessing pipeline started")
log_info(" Timestamp    : {ts()}")
log_info(" Resolution   : {resolution} bp")
log_info(" Chromosomes  : {paste(chr_list, collapse=', ')}")
log_info(" Log file     : {log_file}")
log_info("============================================================")
n_workers <- max(1L, future::availableCores() - 1L)
plan(multisession, workers = n_workers)
log_info("[{ts()}] Parallel backend: {n_workers} workers")
metadata_path <- file.path(dir_base, "ChIP_metadata_20250804.csv")
if (!file.exists(metadata_path)) {
  stop(sprintf("[%s] Metadata file not found: %s", ts(), metadata_path))
}
metadata_files <- fread(metadata_path, header = TRUE)
list_files     <- metadata_files[!get(names(metadata_files)[6]) %in% 'old', get(names(metadata_files)[5])]
discard_files  <- metadata_files[get(names(metadata_files)[6]) %in% 'old', get(names(metadata_files)[5])]
log_info("[{ts()}] {length(discard_files)} file(s) marked as 'old' -> discarded.")
for (j in seq_along(discard_files)) {
  log_info("[{ts()}]   Discard [{j}/{length(discard_files)}]: {discard_files[j]}")
}
log_info("[{ts()}] {length(list_files)} file(s) queued for preprocessing.")
process_one_file <- function(chip_file, chr_vec, res, dir_base) {
  full_path <- file.path(dir_base, "ChIP.GRCh38", chip_file)
  if (!file.exists(full_path)) {
    log_warn("[{ts()}] File not found, skipping: {full_path}")
    return(data.table())
  }
  epi_mark <- strsplit(chip_file, "\\.")[[1]][2]
  bw <- BigWigFile(full_path)
  si <- seqinfo(bw)
  dt_list <- lapply(chr_vec, function(chr) {
    if (!(chr %in% seqnames(si))) return(NULL)
    chr_len <- seqlengths(si)[chr]
    n_full  <- ceiling(chr_len / res)
    if (n_full == 0) return(NULL)
    gr_chr <- GRanges(seqnames = chr,
                      ranges   = IRanges(start = 1, end = chr_len))
    rle_chr <- import(full_path, which = gr_chr, as = "RleList")
    if (is.null(rle_chr) || length(rle_chr) == 0 ||
        !(chr %in% names(rle_chr))) {
      log_warn(sprintf("[%s] No data for %s in %s, filling with NA",
                       ts(), chr, chip_file))
      return(data.table(chrom = chr, bin = seq_len(n_full),
                        value = rep(NA_real_, n_full), mark = epi_mark))
    }
    vals_vec <- as.numeric(rle_chr[[chr]])
    n_use    <- n_full * res
    if (length(vals_vec) < n_use)
      vals_vec <- c(vals_vec, rep(NA_real_, n_use - length(vals_vec)))
    vals_vec <- vals_vec[1:n_use]
    bin_means <- rowMeans(matrix(vals_vec, ncol = res, byrow = TRUE),
                          na.rm = TRUE)
    data.table(chrom = chr, bin = seq_len(n_full),
               value = bin_means, mark = epi_mark)
  })
  rbindlist(dt_list, use.names = TRUE, fill = TRUE)
}
batch_size  <- 10L
n_files     <- length(list_files)
n_batches   <- ceiling(n_files / batch_size)
batch_starts <- seq(1L, n_files, by = batch_size)
log_info("============================================================")
log_info("[{ts()}] Starting batch processing: {n_batches} batch(es), ",
         "{batch_size} files per batch")
log_info("============================================================")
global_start_time <- Sys.time()
for (b in seq_along(batch_starts)) {
  batch_start_time <- Sys.time()
  idx_start <- batch_starts[b]
  idx_end   <- min(idx_start + batch_size - 1L, n_files)
  batch_idx <- idx_start:idx_end
  log_info("")
  log_info("+------------------------------------------+")
  log_info("| Batch {b}/{n_batches}  (files {idx_start}-{idx_end})       | {ts()}")
  log_info("+------------------------------------------+")
  lf  <- list_files
  nf  <- n_files
  tsf <- ts
  pof <- process_one_file
  cl  <- chr_list
  res <- resolution
  db  <- dir_base
  results <- future_map(batch_idx, function(i) {
    chip_file <- lf[i]
    log_info(sprintf("[%s]   [%d/%d] > %s", tsf(), i, nf, chip_file))
    f_start <- Sys.time()
    dt <- pof(chip_file, cl, res, db)
    f_elapsed <- as.numeric(difftime(Sys.time(), f_start, units = "secs"))
    log_info(sprintf("[%s]   [%d/%d] v %s  (%d rows, %.1fs)", tsf(), i, nf, chip_file, nrow(dt), f_elapsed))
    dt
  }, .options = furrr_options(seed = TRUE))
  batch_dt <- rbindlist(results, use.names = TRUE, fill = TRUE)
  if (nrow(batch_dt) == 0) {
    log_warn("[{ts()}] Batch {b} produced no data, skipping.")
    next
  }
  dup_marks <- batch_dt[, .N, by = .(chrom, bin, mark)][N > 1]
  if (nrow(dup_marks) > 0) {
    log_warn("[{ts()}] Duplicate mark entries: {nrow(dup_marks)} rows collapsed by mean.")
    batch_dt <- batch_dt[, .(value = mean(value, na.rm = TRUE)), by = .(chrom, bin, mark)]
  }
  wide_dt   <- dcast(batch_dt, chrom + bin ~ mark, value.var = "value")
  chr_levels <- c(paste0("chr", 1:22), "chrX", "chrY")
  wide_dt[, chrom := factor(chrom, levels = chr_levels)]
  setorder(wide_dt, chrom, bin, na.last = TRUE)
  wide_dt[, bin := seq_len(.N)]
  mark_cols <- setdiff(names(wide_dt), c("chrom", "bin"))
  setcolorder(wide_dt, c("bin", "chrom", mark_cols))
  out_path  <- file.path(save_base, paste0("ChIP.", b, ".csv"))
  fwrite(wide_dt, out_path)
  batch_elapsed <- as.numeric(difftime(Sys.time(), batch_start_time, units = "secs"))
  log_info("[{ts()}] v Batch {b} written -> {out_path}")
  log_info("           Rows: {format(nrow(wide_dt), big.mark=',')}  |  Marks: {ncol(wide_dt)-2}  |  Time: {fmt_duration(batch_elapsed)}")
  global_elapsed <- as.numeric(difftime(Sys.time(), global_start_time, units = "secs"))
  avg_per_batch  <- global_elapsed / b
  remaining      <- avg_per_batch * (n_batches - b)
  eta_time       <- Sys.time() + remaining
  pct_done       <- round(b / n_batches * 100, 1)
  log_info("           Progress: {b}/{n_batches} batches ({pct_done}%)  |  Elapsed: {fmt_duration(global_elapsed)}  |  ETA: {format(eta_time, '%Y-%m-%d %H:%M:%S')} ({fmt_duration(remaining)} remaining)")
}
global_elapsed <- as.numeric(difftime(Sys.time(), global_start_time, units = "secs"))
log_info("")
log_info("============================================================")
log_info("[{ts()}] All done!")
log_info("============================================================")
log_info("  Total files processed   : {n_files}")
log_info("  Total batches written   : {n_batches}")
log_info("  Output directory        : {save_base}")
log_info("  Total wall-clock time   : {fmt_duration(global_elapsed)}")
log_info("============================================================")
plan(sequential)
