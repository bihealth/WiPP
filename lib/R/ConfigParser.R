library(ini)
library(R6)


##############################
##
## The ConfigParser
##
##############################

##' R6 object to parse an INI file
##'
##' This class creates an object that represents an INI file. It
##' also supports interpolation of variables of the form \code{\%(...)s}.
##' More detail on how the interpolation works in given in the
##' explanation of method \code{get}.
##' @importFrom ini read.ini
##' @import R6
##' @export
ConfigParser <- R6Class(
    classname="configParser",
    public=list(
        initialize=function(init=NULL, optionxform=tolower) {
            " Initializes the object"
            ""
            "Initializes the object, optionally with pre-set variables given"
            "as a list or an environment"
            "@param init A named character vector, named list or an environment of variables to pre-set (the will be"
            "put into the \\code{DEFAULT} section)"
            "@param optionxform Function through which all option and section names are funneled. By default \\code{tolower}"
            "to make all names lowercase. To switch off, pass in \\code{identity}"

            self$optionxform <- optionxform
            if(!is.null(init)) {
                if(is.environment(init)) {
                    init <- as.list(init)
                }
                if(is.character(init)) {
                    init <- as.list(init)
                }
                if(is.list(init)) {
                    ## check that the the list is named
                    if(is.null(names(init)) || any(names(init) == "")) {
                        stop("All elements of init need to be named")
                    }
                    names(init) <- self$optionxform(names(init))
                    self$data[[self$optionxform("DEFAULT")]] <- init
                }
                else {
                    stop("init has to be a list or an environment")
                }
            }
        },
        read=function(filepath, encoding = getOption("encoding")) {
            "Read in an INI file"
            ""
            "Reads the INI given into the object. It will be added to the"
            "internal storage of the object. If applying these functions"
            "several times, later definitions overwrite those that"
            "were read earlier."
            "@param filepath The path to the INI file."
            "@param encoding The encoding to use"
            "@return The object, but with the added data of the file"
            
            ini_input <- ini::read.ini(filepath=filepath, encoding=encoding)
            
            ## go through and add the values
            ## use the transform function and ensure that when transform causes
            ## duplication, only the last value is being used
            for(i in seq_along(ini_input)) {
                ## ensure that also the section name is transformed
                section_name <- self$optionxform(names(ini_input)[i])
                
                ## combine the old data and the new for the same section
                ## ensure that the old is first, so that it is overwriten
                single_section <- c(self$data[[section_name]], ini_input[[i]])
                names(single_section) <- self$optionxform(names(single_section))

                ## have to reverse in this line so that the last occurence is picked up
                single_section <- rev(single_section)[unique(names(single_section))]
                self$data[[section_name]] <- single_section
            }
            return(invisible(self))
        },
        write=function(filepath, encoding = getOption("encoding")) {
            "Write an INI file"
            ""
            "Write the ConfigParser object into the INI file. It writes all the different sections"
            "using uninterpolated variables."
            "@param filepath The path to the INI file."
            "@param encoding The encoding to use"
            ""
            "@return The ConfigParser object itself"

            ini::write.ini(self$data, filepath=filepath, encoding=encoding)
            return(invisible(self))
        },
        get=function(option, fallback, section="default", interpolate=TRUE) {
            "Get the value of a option"
            ""
            "@param option The option for which to get the value"
            "@param fallback The fallback value to return if there is no value for the option. If missing,"
            "an error will be thrown if the option has no value."
            "@param section The section (or several) from which to read the option. It will try to read the"
            "option in the given section from first to last, with \\code{DEFAULT} always being the"
            "last"
            "@param interpolate Should the values be interpolated. This will try to replace variables of"
            "the form \\code{\\%(...)s}"
            "@return The value of the option"            

            ## we want to do interpolation
            ## any values to be interpolated have to be of the form %(dir)s
            option <- self$optionxform(option)
            
            # get section name and default section name
            section <- c(section, "DEFAULT")
            section <- unique(self$optionxform(section))

            ## search through all the section until the right one is found
            optionFound <- FALSE
            section_found <- NULL
            for(i in seq_along(section)) {
                single_section <- self$data[[section[i]]]

                value <- single_section[[option]]
                if(!is.null(value)) {
                    optionFound <- TRUE
                    sections_remain <- section[i:length(section)]
                    break
                }
            }
            if(!optionFound) {
                if(missing(fallback)) {
                    custom_stop("NoOptionError", "Option not found and no fallback given")
                }
                else {
                    return(fallback)
                }
            }

            ## now that we have the option, we have to look into the
            ## possible necessary interpolation
            if(interpolate) {
                while(length(ioption <- self$optionxform(interpolation_option(value))) > 0) {
                    ## need to get the value of the option
                    ## ensure you look in the current section first
                    if(ioption == option) {
                        ## can't look in the current section for yourself
                        if(length(sections_remain) == 1) {
                            stop("Recursion of option '", option, "' in section '", sections_remain, "'")
                        }
                        option_value <- self$get(option=ioption, fallback=NA, section=sections_remain[-1], interpolate=TRUE)
                    }
                    else {
                        option_value <- self$get(option=ioption, fallback=NA, section=sections_remain, interpolate=TRUE)
                    }
                    ## if NA is being returned, the option was not found
                    if(is.na(option_value)) {
                        stop("The option '", ioption, "' needed for interpolation could not be found")
                    }
                    value <- do_replacement(value, option=ioption, replacement = option_value)
                }
            }
            return(value)
        },
        getboolean=function(option, fallback, section="default", interpolate=TRUE) {
            "Same as \\code{get}, but results coerced to a logical."
            
            res <- try(self$get(option=option, section=section, interpolate=interpolate))
            if(inherits(res, "try-error")) {
                # no value found
                if(missing(fallback)) {
                    custom_stop("NoOptionError", "Option not found and no fallback given")
                }
                else {
                    return(fallback)
                }
            }
            else {
                return(as.logical(res))
            }
        },
        getfloat=function(option, fallback, section="default", interpolate=TRUE) {
            "Same as \\code{get}, but the result coerced to a float."
            
            res <- try(self$get(option=option, section=section, interpolate=interpolate))
            if(inherits(res, "try-error")) {
                # no value found
                if(missing(fallback)) {
                    custom_stop("NoOptionError", "Option not found and no fallback given")
                }
                else {
                    return(fallback)
                }
            }
            else {
                return(as.numeric(res))
            }
        },
        
        set=function(option, value, section, error_on_new_section=TRUE) {
            "Set option to a value in a section"
            ""
            "Sets an option to the given value (which can include variables for"
            "interpolation) in the given section."
            "@param option Name of the option to set (can be a character vector)"
            "@param value Value of the option (same length as \\code{option})"
            "@param section Character vector of length 1"
            "@param error_on_new_section Should an error be raised if the section does"
            "not exist yet"
            "@return Return of the adjusted \\code{ConfigParser} object itself" 

            option <- self$optionxform(as.character(option))
            values <- as.character(value)
            section <- as.character(section)

            if(length(section) != 1) {
                stop("section needs to be of length 1")
            }
            if(length(option) != length(value)) {
                stop("'option' and 'value' need to be of same length")
            }
            if(!(section %in% names(self$data))) {
                if(error_on_new_section) {
                    stop("Section ", section, " does not exist")
                }
                else {
                    self$data[[self$optionxform(section)]] <- list()
                }
            }

            ## write the data into the list
            new_list <- setNames(as.list(value), nm=option)
            new_list <- c(new_list, self$data[[self$optionxform(section)]])
            new_list[unique(names(new_list))]
            self$data[[self$optionxform(section)]] <- new_list
            
            return(invisible(self))
        },
        data=list(),
        optionxform=NULL
    )
)
