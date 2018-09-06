#!/usr/bin/env Rscript


# Rewrite mzR import function to allow import without metaData
new_netCDFInstrumentInfo <- function(ncid) {
  return(list(
    model = '', manufacturer = '', ionisation = '', detector = '', analyzer = NA
  ))
}
assignInNamespace('netCDFInstrumentInfo', new_netCDFInstrumentInfo, getNamespace('mzR'))


# Combines grouped pseudospectra from CAMERA to peaks
combine_CAMERA_peaks <- function(input, feature_list) {
  peak <- feature_list[input,]
  # mz_str <- construct_mz_str(peak$mz, )
  return(c(
    mean(peak$rt),
    mean(peak$rtmin),
    mean(peak$rtmax),
    paste(unique(paste(peak$mz, peak$maxo, sep=':')), collapse = ',')
  ))
}


# Get the first and last time point from a spectra (Rt_min, Rt_max)
get_Rt_minMax <- function(profile_data) {
  data_str <- strsplit(profile_data, ' ')
  # First Intensity
  Rt_min <- strsplit(data_str[[1]][1], ',', fixed=TRUE)[[1]][1]
  # Last Intensity
  Rt_max <- strsplit(data_str[[1]][length(data_str[[1]])], ',', fixed=TRUE)[[1]][1]
  return(c(as.numeric(Rt_min), as.numeric(Rt_max)))
}

# Add Rt_min, Rt_max to eRah peaks on individual samples
add_Rt_MinMax <- function(deconv_data) {
  # Get vector of Rt: c(Rt_min, Rt_max)
  Rt_borders <- sapply(as.character(deconv_data$Profile), get_Rt_minMax, USE.NAMES=FALSE)
  deconv_data$Rt_min <- Rt_borders[1,]
  deconv_data$Rt_max <- Rt_borders[2,]
  return(deconv_data)
}

# Add to default eRah output the Rt_min and Rt_max values
add_Rt_minMax <- function(sample, alignID) {
  if(alignID %in% sample$AlignID) {
    peak_idx <- match(alignID, sample$AlignID)
    return(c(sample[peak_idx,]$Rt_min, sample[peak_idx,]$Rt_max))
  } else {
    return(c(NA, NA))
  }
}

prepare_eRah_output <- function(peak, single_data) {
  rt_df <- sapply(single_data, add_Rt_minMax, alignID=as.numeric(peak[['AlignID']]), USE.NAMES=FALSE)
  rt_mean  <- apply(rt_df, 1, mean, na.rm=TRUE)
  # Change unit from mins to secs
  peak[['rtmin']] <- rt_mean[[1]] * 60
  peak[['rtmax']] <- rt_mean[[2]] * 60
  peak[['tmean']] <- as.numeric(peak[['tmean']]) * 60
  return(peak)
}