# WiPP - A **W**orkflow for **i**mproved **P**eak **D**etection

## Requirements
- conda
- libnetcdf11 (required by R package CAMERA)

## Installation
```bash
git clone https://github.com/bihealth/wipp_dev.git
cd wipp_dev
make
```

## Running of test project

1. Change in directory
```bash
cd ./projects/example_project
```
2. Call training data generation with n nodes (example: n = 4)
```bash
../../run_WiPP.sh tr -n 4 
```
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
