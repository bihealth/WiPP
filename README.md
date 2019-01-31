# WiPP - A **W**orkflow for **i**mproved **P**eak **D**etection
**WiPP** is an open source large scale GC-MS data preprocessing workflow built in Python 3 that uses machine learning to optimise, automate and combine the peak detection process of commonly used peak picking algorithms.

**WiPP** has been developed as a collaborative effort between the Berlin Institute of Health (BIH) Metabolomics platform, the BIH Core Unit Bioinformatics, the INRA Plateforme d'Exploration du Métabolisme, and the INRA Laboratoire d'Etude des Résidus et Contaminants dans les Aliments.

This document aims to help you get started with **WiPP** and brings through the minimum requirements to install, set up, and run the software. However, we strongly recommend you to read through the [complete user guide](documentation/USERGUIDE.md) for a full and advanced use of **WiPP**.

## Requirements
- conda ([Bioconda website](https://bioconda.github.io/))
- libnetcdf11 ([Ubuntu packages website](https://packages.ubuntu.com/xenial/libs/libnetcdf11))

## Installation
You can install **WiPP** using the following command:
```bash
git clone https://github.com/bihealth/wipp_dev.git
cd wipp_dev
make
```
Now you are ready to run **WiPP**!


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
4. Call peak picking with n nodes (example: n = 4)
```bash
../../run_WiPP.sh pp -n 4 
```
## Further details
TODO <YG, 2018.09.06> Update
