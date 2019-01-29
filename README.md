# WiPP - A **W**orkflow for **i**mproved **P**eak **D**etection
Some introduction text \
TODO <YG, 2018.09.06> Update

## Requirements
- conda
- libnetcdf11

## Installation
```bash
git clone https://github.com/bihealth/wipp_dev.git
cd wipp_dev
make
```

## Running of test project

1. Change into example_project directory
```bash
cd ./projects/example_project
```
2. Call training data generation with n nodes (example: n = 4)
```bash
../../run_WiPP.sh tr -n 4 
```
(Note: Running this for the first time takes a while: another conda environment is created)

3. Call peak annotation
```bash
../../run_WiPP.sh an
```
To use an already trained example classifier, uncomment the following line in the *config.yaml* file:
```
# path: example_classifier
```

4. Call peak picking with n nodes (example: n = 4)
```bash
../../run_WiPP.sh pp -n 4 
```
## Troubleshooting
- *there is no package called ‘MSnbase’*

Add the following line **to the top** of the file "envs/R_env.yaml" under the section "dependencies":
```
- bioconductor-msnbase=2.4.0=r341h470a237_1
```

## Further details
TODO <YG, 2018.09.06> Update
