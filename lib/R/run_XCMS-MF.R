#!/usr/bin/env Rscript

library(argparser)
library(ConfigParser)
suppressMessages(library(CAMERA))

#-------------------------------------------------------------------------------
# ARGUMENT PARSER
#-------------------------------------------------------------------------------
p <- arg_parser('Run XCMS matchedFilter on sample group with specified parameters')
p <- add_argument(p, 'in_path', help = 'input dir path (str)')
p <- add_argument(p, 'out_path', help = 'output dir path (str)')
p <- add_argument(p, '--fwhm',
  default = 5, help = 'full width half maximum (float)'
)
p <- add_argument(p, '--sn',
  default = 10, help = 'signal/noise ratio cutoff (float)'
)
p <- add_argument(p, '--mzdiff',
  default = 'm0.001',
  help = 'minimum difference in m/z dimension for peaks 
    with overlapping retention times (float)'
)
p <- add_argument(p, '--step',
  default = 0.1, help = 'width of the bins/slices in m/z dimension'
)
p <- add_argument(p, '--steps',
  default = 2, help = 'number of bins to be merged before filtration (int)'
)

p <- add_argument(p, '--config',
  default = '../../config/XCMS-MF_default.INI',
  help = 'config dir path (default: config/XCMS-MF_default.INI)'
)
p <- add_argument(p, '--groupCorr',
  flag = TRUE, help = 'Additional grouping of features by correlation'
)
p <- add_argument(p, '--cores', default = 1, help = 'number of cores (int)')
p <- add_argument(p, '--utils',
  default = '/home/nborgsmu/Desktop/Masterarbeit/PeakDetection_dev/lib/R/utils.R',
  help = 'path to ultis.R'
)


argv <- parse_args(p)
source(argv$utils)

#-------------------------------------------------------------------------------
# CONFIG PARSER
#-------------------------------------------------------------------------------
config <- ConfigParser$new()
config$read(argv$config)

#-------------------------------------------------------------------------------
# DATA IMPORT/EXPORT STRUCTURE
#-------------------------------------------------------------------------------
in_path <- argv$in_path

if (file_test('-f', in_path)) {
  files = c(in_path)
} else {
  files <- list.files(
    path=in_path,
    pattern='*.[cdf|CDF|mzML|mzml|mzData|mzdata]',
    all.files=TRUE,
    full.names=TRUE,
    recursive=TRUE,
    ignore.case=TRUE
  )
}
out_path <- argv$out_path

#-------------------------------------------------------------------------------
# CAMERA
#-------------------------------------------------------------------------------

MF.features <- xcmsSet(
  files,
  method = "matchedFilter",
  fwhm = argv$fwhm,
  snthresh = argv$sn,
  step = argv$step,
  steps = argv$steps,
  mzdiff = as.numeric(gsub('m', '-', argv$mzdiff)),
  max = config$getfloat('max', 500, 'matchedFilter')
)

# saveRDS(MF.features, file=gsub('.csv', '.rds', out_path))
# readRDS() for loading Object

MF.features.an <- xsAnnotate(
  MF.features,
  sample = c(1:length(MF.features)), # automatic selection. For all samples: NA
  nSlaves = argv$cores,
  polarity = 'positive' # Always for EI
)

# Group peaks according to there retention time into pseudospectra-groups.
# The FWHM (full width at half maximum) of a peak is estimated as: FWHM=SD*2.35.
# For  the calculation of the SD, the peak is assumed as normal distributed.
MF.features.an <- groupFWHM(
  MF.features.an,
  sigma = config$getfloat('sigma', 6, 'groupFWHM'),
  perfwhm = config$getfloat('perfwhm', 1, 'groupFWHM'), # same time point = Rt_med +/- FWHM * perfwhm
  intval = config$get('intval', 'maxo', 'groupFWHM')
)

# Calculates the pearson correlation coefficient based on the peak shapes of 
# every peak in the pseudospectrum to separate co-eluted substances
if (argv$groupCorr) {
  MF.features.an <- groupCorr(
    MF.features.an,
    cor_eic_th = config$getfloat('cor_eic_th', 0.75, 'groupCorr'),
    pval = config$getfloat('pval', 0.05, 'groupCorr'),
    graphMethod = config$get('graphMethod', 'hcs', 'groupCorr'),
    calcIso = config$getboolean('calcIso', FALSE, 'groupCorr'),
    calcCiS = config$getboolean('calcCis', TRUE, 'groupCorr'),
    calcCaS = config$getboolean('calcCas', FALSE, 'groupCorr'),
    cor_exp_th = config$getfloat('cor_exp_th', 0.75, 'groupCorr'),
    intval = config$get('intval', 'into', 'groupCorr')
  )
}

# TODO <NB, 18.01.16> What about 'groupDen' (XCMS) grouping algorithm?!


#-------------------------------------------------------------------------------
# Output
#-------------------------------------------------------------------------------
features <- getPeaklist(MF.features.an)
psspec <- MF.features.an@pspectra

# Combine single peak data to one list
df_raw <- lapply(psspec, combine_CAMERA_peaks, feature_list=features)
# Convert nested list to matrix
df <- matrix(unlist(df_raw), ncol = 4, byrow=TRUE)
# Rename columns
colnames(df) <- c('rt', 'rtmin', 'rtmax', 'mz')
# Save as tsv table
write.table(df, out_path, sep="\t", row.names = FALSE)
