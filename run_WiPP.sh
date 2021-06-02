#!/bin/bash

#-------- By default jobs will run locally using 4 CPUs (4 nodes) --------
HPC=false
NODES=4
GIGS_PER_CPU=6
CONFIG="$PWD/config.yaml"
SNAKE="default"
STEP_HAS_PARAMS=true

WIPP_PATH="$(dirname $(realpath -s $0))"
LOCAL_WIPP_STEP_EXE="$WIPP_PATH"

WIPP_MODE=$1
if [ "$WIPP_MODE" == "tr" ]
then
    echo 'Running WiPP - Training'
    LOCAL_WIPP_STEP_EXE+="/bin/Snakefile_tr"
    LOCAL_WIPP_STEP_EXE="snakemake -s $LOCAL_WIPP_STEP_EXE --use-conda -k"
elif [ "$WIPP_MODE" == "pp" ]
then
    echo 'Running WiPP - Peak Picking'
    LOCAL_WIPP_STEP_EXE+="/bin/Snakefile_pp"
    LOCAL_WIPP_STEP_EXE="snakemake -s $LOCAL_WIPP_STEP_EXE --use-conda -k"
elif [ "$WIPP_MODE" == "an" ]
then
    echo 'Running WiPP - Annotation'
    LOCAL_WIPP_STEP_EXE+="/lib/python/peak_annotation.py"
    LOCAL_WIPP_STEP_EXE="python3 "$LOCAL_WIPP_STEP_EXE
    STEP_HAS_PARAMS=false # additional params and config are not applicable for this step
    echo 'NOTE: This step will always be run locally on a single node, and must be started from a terminal with X11 forwarding'
else
    echo 'Unknown Mode: $WIPP_MODE. Allowed are: tr|an|pp'
    exit 1
fi
shift

if $STEP_HAS_PARAMS
then
    while [ "$1" != "" ]; do
        case $1 in
            -c | --config )         shift
                                    LOCAL_WIPP_STEP_EXE+=" --configfile "
                                    LOCAL_WIPP_STEP_EXE+="$PWD/$1"
                                    CONFIG="$PWD/$1"
                                    ;;
            -n | --nodes )          shift
                                    LOCAL_WIPP_STEP_EXE+=" --cores "
                                    LOCAL_WIPP_STEP_EXE+=$1
                                    NODES=$1
                                    ;;
            -g | --gigs-per-cpu )   shift
                                    GIGS_PER_CPU=$1
                                    ;;
            -x | --external )       shift
                                    HPC=true
                                    ;;
            -s | --snake )          shift
                                    LOCAL_WIPP_STEP_EXE+=" "
                                    LOCAL_WIPP_STEP_EXE+=$1
                                    SNAKE=$1
                                    ;;
         esac
        shift
    done
fi

if $HPC
then
    echo "Checking conda version (cluster jobs require version 4.4.0 - 4.9.2)"
    REQ_MAJOR_VER=4
    MIN_REQ_MINOR_VER=4
    MAX_REQ_MINOR_VER=9
    
    USER_CONDA_VER=$(conda --version | cut -f 2 -d" ")
    USER_MAJOR_VER=$(conda --version | cut -f 2 -d" " | cut -f 1 -d".")
    USER_MINOR_VER=$(conda --version | cut -f 2 -d" " | cut -f 2 -d".")
    echo "You are running conda version ${USER_CONDA_VER}"

    if [[ $USER_MAJOR_VER -eq $REQ_MAJOR_VER ]]
    then
        if [[ $USER_MINOR_VER -lt $MIN_REQ_MINOR_VER ]]
        then
            echo "Please upgrade to conda version 4.9.2 to run WiPP"
            exit 0
        else # minor >= min supported
            if [[ $USER_MINOR_VER -gt $MAX_REQ_MINOR_VER ]]
            then
                echo "Please downgrade to conda version 4.9.2 to run WiPP"
                exit 0
            else # minor <= max supported
                echo "Your conda version is OK"
            fi
        fi
    else
        echo "Please install conda version 4.9.2 to run WiPP"
        exit 0
    fi

    # -- Create output dir for the slurm run logs, if it does not already exist --
    if [ ! -d slurm_log ]
    then
        mkdir -p slurm_log
    fi

    # -- Submit job(s) to SLURM scheduler --
    PRJ_DIR=$(realpath .)
    PRJ_NAME=$(basename $PRJ_DIR)
    CMD="sbatch --job-name=wipp_${WIPP_MODE}_${PRJ_NAME} --cpus-per-task=${NODES} --mem-per-cpu=${GIGS_PER_CPU}G --export=WIPP_PATH=${WIPP_PATH},NODES=${NODES},CONFIG=${CONFIG},WIPP_MODE=${WIPP_MODE},SNAKE=${SNAKE},PRJ_DIR=${PRJ_DIR} ${WIPP_PATH}/bin/general_wipp_slurm_job.sh"
    echo 'Submitting cluster job:'
    echo "\"$CMD\""
    $CMD
else
    # -- Run job locally --
    # WiPP requires conda version <= 4.9.2 and >= 4.3.X
    REQ_MAJOR_VER=4
    MIN_REQ_MINOR_VER=3
    MAX_REQ_MINOR_VER=9

    USER_CONDA_VER=$(conda --version | cut -f 2 -d" ")
    USER_MAJOR_VER=$(conda --version | cut -f 2 -d" " | cut -f 1 -d".")
    USER_MINOR_VER=$(conda --version | cut -f 2 -d" " | cut -f 2 -d".")
    echo "You are running conda version ${USER_CONDA_VER}"

    if [[ $USER_MAJOR_VER -eq $REQ_MAJOR_VER ]]
    then
        if [[ $USER_MINOR_VER -eq $MIN_REQ_MINOR_VER ]]
        then
            # use source activate for conda version 4.3.X
            echo 'Activating WiPP conda env (using syntax for version < 4.4)'
            source activate WiPP

        elif [[ $USER_MINOR_VER -lt $MIN_REQ_MINOR_VER ]]
        then
            echo "Please upgrade to conda version 4.9.2 to run WiPP"
            exit 0
        else # minor >= min supported
            if [[ $USER_MINOR_VER -gt $MAX_REQ_MINOR_VER ]]
            then
                echo "The WiPP R env uses python 3.6.5 and conda 4.10.X formally drops support for python 3.6"
                echo "Please downgrade to conda version 4.9.2 to run WiPP"
                exit 0
            else # all is good
                # initialize conda and use conda activate for versions >= 4.4
                echo 'Activating WiPP conda env'
                eval "$(conda shell.bash hook)"        
                conda activate WiPP
            fi
        fi
    else
        echo "Please install conda version 4.9.2 to run WiPP"
        exit 0
    fi

    # Set environment variable(s)
    export WIPP_PATH="$WIPP_PATH"

    # Run workflow
    if $STEP_HAS_PARAMS
    then
        # Check config 
        echo ''
        echo 'Checking config:'
        CONFIG_CHECK="$WIPP_PATH"
        CONFIG_CHECK+="/lib/python/utils.py"

        python3 "$CONFIG_CHECK" -c "$CONFIG" -t "$WIPP_MODE"

        echo 'Running Snakemake:'
    else
        echo 'Running Python:'
    fi
    echo "$LOCAL_WIPP_STEP_EXE"
    $LOCAL_WIPP_STEP_EXE
fi
