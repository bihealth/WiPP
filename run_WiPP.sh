#!/bin/bash

WIPP_PATH="`dirname \"$0\"`"
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
elif [ "$WIPP_MODE" == "ann" ]
then
    ANNOTATION_EXE="$WIPP_PATH"
    ANNOTATION_EXE+="/lib/python/peak_annotation.py"
    python3 "$ANNOTATION_EXE"
    exit
else
    echo 'Unknown Mode: $WIPP_MODE. Allowed are: tr|pp'
    exit 1
fi
shift

cores=1
config=

while [ "$1" != "" ]; do
    case $1 in
        -c | --config )         shift
                                config="--configfile $1"
                                ;;
        -n | --nodes )          shift
                                cores=$1
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
snakemake -s "$SNAKEMAKE_EXE" --use-conda -k --cores "$cores" --configfile "$config"