#!/usr/bin/env Rscript

library(argparser)
suppressWarnings(suppressMessages(library(CAMERA)))

#-------------------------------------------------------------------------------
# ARGUMENT PARSER
#-------------------------------------------------------------------------------
p <- arg_parser('Run XCMS centwave on sample group with specified parameters')
p <- add_argument(p, 'in_path', help = 'input dir path (str)')
p <- add_argument(p, 'out_path', help = 'output dir path (str)')
p <- add_argument(p, '--peakwidth',
  default = c(5, 20), nargs = 2, help = 'peakwidth parameters: min max'
)
p <- add_argument(p, '--mzdiff',
  default = 'm0.001',
  help = 'minimum difference in m/z dimension for peaks 
    with overlapping retention times (float)'
)
p <- add_argument(p, '--ppm',
  default = 25,
  help = 'mass accuracy tolerance in parts per million (int)'
)
p <- add_argument(p, '--param_file', default = '',
  help = 'path to a csv file containing algorithm parameters.'
)
p <- add_argument(p, '--config',
  default = '../../pp_configs/XCMS-CW_default.INI',
  help = 'config dir path (default: config/XCMS-CW_default.INI)'
)
p <- add_argument(p, '--groupCorr',
  flag = TRUE, help = 'Additional grouping of features by correlation'
)
p <- add_argument(p, '--cores', default = 1, help = 'number of cores (int)')
p <- add_argument(p, '--utils',
  default = '../../lib/R/utils.R',
  help = 'path to ultis.R'
)

argv <- parse_args(p)
source(argv$utils)
source('../../lib/R/ConfigParser.R')

#-------------------------------------------------------------------------------
# CONFIG PARSER
#-------------------------------------------------------------------------------
config <- ConfigParser$new()
config$read(argv$config)

if (argv$param_file != '') {
  params <- read.table(argv$param_file)
  for (row in 1:dim(params)[1]) {
    if (params[[1]][[row]] == 'par0') {
      argv$peakwidth[1] <- params[[2]][[row]]
    } else if (params[[1]][[row]] == 'par1') {
      argv$peakwidth[2] <- params[[2]][[row]]
    } else if (params[[1]][[row]] == 'par2') {
      argv$mzdiff <- params[[2]][[row]]
    } else if (params[[1]][[row]] == 'par3') {
      argv$ppm <- params[[2]][[row]]
    }
  }
}

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
CW.features <- xcmsSet(
  files,
  method = "centWave",
  peakwidth = argv$peakwidth,
  ppm = argv$ppm,
  snthresh = config$getfloat('snthresh', 0, 'centWave'),
  prefilter = eval(parse(text=config$get('prefilter', 'c(3, 100)', 'centWave'))),
  mzCenterFun =  config$get('mzCenterFun', 'wMean', 'centWave'),
  integrate = config$getfloat('integrate', 1, 'centWave'),
  mzdiff = as.numeric(gsub('m', '-', argv$mzdiff)),
  fitgauss = config$getboolean('fitgauss', FALSE, 'centWave'),
  noise = config$getfloat('noise', 0, 'centWave'),
)

CW.features.an <- xsAnnotate(
  CW.features,
  sample = c(1:length(CW.features)), # all samples. For automatic selection: NA
  nSlaves = argv$cores,
  polarity = 'positive' # Always for EI
)

# Group peaks according to there retention time into pseudospectra-groups.
# The FWHM (full width at half maximum) of a peak is estimated as: FWHM=SD*2.35.
# For  the calculation of the SD, the peak is assumed as normal distributed.
CW.features.an <- groupFWHM(
  CW.features.an,
  sigma = config$getfloat('sigma', 6, 'groupFWHM'),
  perfwhm = config$getfloat('perfwhm', 1, 'groupFWHM'), # same time point = Rt_med +/- FWHM * perfwhm
  intval = config$get('intval', 'maxo', 'groupFWHM')
)

# Calculates the pearson correlation coefficient based on the peak shapes of 
# every peak in the pseudospectrum to separate co-eluted substances
if (argv$groupCorr) {
  CW.features.an <- groupCorr(
    CW.features.an,
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

#-------------------------------------------------------------------------------
# Output
#-------------------------------------------------------------------------------
features <- getPeaklist(CW.features.an)
psspec <- CW.features.an@pspectra

# Combine single peak data to one list
df_raw <- lapply(psspec, combine_CAMERA_peaks, feature_list=features)
# Convert nested list to matrix
df <- matrix(unlist(df_raw), ncol = 4, byrow=TRUE)
# Rename columns
colnames(df) <- c('rt', 'rtmin', 'rtmax', 'mz')
# Save as tsv table
write.table(df, out_path, sep="\t", row.names = FALSE)