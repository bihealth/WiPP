#!/bin/bash

#-------- By default jobs will run locally on a single node --------
HPC=false
NODES=1
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
    # create output dir for the slurm run logs, if it does not already exist
    if [ ! -d slurm_log ]
    then
        mkdir -p slurm_log
    fi

    # -- Submit job(s) to SLURM scheduler --
    PRJ_DIR=$(realpath .)
    PRJ_NAME=$(basename $PRJ_DIR)
    CMD="sbatch --job-name=wipp_${WIPP_MODE}_${PRJ_NAME} --cpus-per-task=${NODES} --export=WIPP_PATH=${WIPP_PATH},NODES=${NODES},CONFIG=${CONFIG},WIPP_MODE=${WIPP_MODE},SNAKE=${SNAKE},PRJ_DIR=${PRJ_DIR} ${WIPP_PATH}/bin/general_wipp_slurm_job.sh"
    echo 'Submitting cluster job:'
    echo "\"$CMD\""
    $CMD
else
    # -- Run job locally --
    # Note: The logic for activating conda ASSUMES the oldest possible version being used is 4.3.X
    # Activate the conda environment
    if [[ "$(conda --version)" =~ "conda 4.3." ]]
    then
        echo 'Activating WiPP env (for conda version 4.3)'
        source activate WiPP
    else
        # Need to first initialize conda
        eval "$(conda shell.bash hook)"
        echo 'Activating WiPP conda env'
        conda activate WiPP
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
