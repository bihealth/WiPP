#!/bin/bash

# note: sbatch params --cpus-per-task and --job-name are set by the calling script

# #SBATCH --partition=medium
# #SBATCH --time=24:00:00  #max is 6-23:59:00

#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_log/%x-%J.log

set -e
umask ug=rwx,o=
export TMPDIR=${HOME}/scratch/tmp

# -- activate conda and set up env
# Need to first initialize conda
eval "$(conda shell.bash hook)"

echo 'Activating WiPP conda env'
conda activate WiPP

# Set environment variable(s)
export WIPP_PATH=$WIPP_PATH

# Construct the snakemake command
WIPP_STEP_EXE="snakemake -s ${WIPP_PATH}/bin/Snakefile_${WIPP_MODE} --use-conda -k --configfile ${CONFIG} --cores ${NODES}"

if [ ${SNAKE} != "default" ]
then
    WIPP_STEP_EXE+=" ${SNAKE}"
fi

# Check config
CONFIG_CHECKER="${WIPP_PATH}/lib/python/utils.py"
echo ''
echo 'Checking config:'
CHK_CMD="python3 ${CONFIG_CHECKER} -c ${CONFIG} -t ${WIPP_MODE}"
echo $CHK_CMD
$CHK_CMD

# Run workflow
echo 'Writing to project directory:'
echo ${PRJ_DIR}

cd ${PRJ_DIR}
echo ''
echo 'Running snakemake command:'
echo $WIPP_STEP_EXE

$WIPP_STEP_EXE
