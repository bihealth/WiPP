#!/bin/bash

WIPP_PATH="$(dirname $(realpath -s $0))"
SNAKEMAKE_EXE="$WIPP_PATH"
CONFIG_CHECK="$WIPP_PATH"
CONFIG_CHECK+="/lib/python/utils.py"

# start the right environment
source activate WiPP
# Set environment variable(s)
export WIPP_PATH="$WIPP_PATH"

WIPP_MODE=$1
if [ "$WIPP_MODE" == "tr" ]
then
    echo 'Running WiPP - Training'
    SNAKEMAKE_EXE+="/bin/Snakefile_tr"
elif [ "$WIPP_MODE" == "pp" ]
then
    echo 'Running WiPP - Peak Picking'
    SNAKEMAKE_EXE+="/bin/Snakefile_pp"
elif [ "$WIPP_MODE" == "an" ]
then
    ANNOTATION_EXE="$WIPP_PATH"
    ANNOTATION_EXE+="/lib/python/peak_annotation.py"
    python3 "$ANNOTATION_EXE"
    exit
else
    echo 'Unknown Mode: $WIPP_MODE. Allowed are: tr|an|pp'
    exit 1
fi
shift

SNAKE_CMD="snakemake -s $SNAKEMAKE_EXE --use-conda -k"

while [ "$1" != "" ]; do
    case $1 in
        -c | --config )         shift
                                SNAKE_CMD+=" --configfile "
                                SNAKE_CMD+="$PWD/$1"
                                ;;
        -n | --nodes )          shift
                                SNAKE_CMD+=" --cores "
                                SNAKE_CMD+=$1
                                ;;
        -s | --snake )          shift
                                SNAKE_CMD+=" "
                                SNAKE_CMD+=$1
                                ;;
    esac
    shift
done


# Check config
echo ''
echo 'Checking config:'
python3 "$CONFIG_CHECK" -c "$config" -t "$WIPP_MODE"
# Run workflow
echo ''
echo 'Running Snakemake:'
echo "$SNAKE_CMD"
$SNAKE_CMD