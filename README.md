# WiPP - A **W**orkflow for **i**mproved **P**eak **P**ickng
**WiPP** is an open source large scale GC-MS data preprocessing workflow built in Python 3 that uses machine learning to optimise, automate and combine the peak detection process of commonly used peak picking algorithms.

**WiPP** has been developed as a collaborative effort between the Berlin Institute of Health (BIH) Metabolomics platform, the BIH Core Unit Bioinformatics, the INRA Plateforme d'Exploration du Métabolisme, and the INRA Laboratoire d'Etude des Résidus et Contaminants dans les Aliments.

This document aims to help you get started with **WiPP** and brings through the minimum requirements to install, set up, and run the software. However, we strongly recommend you to read through the [complete user guide](documentation/USERGUIDE.md) for a full and advanced use of **WiPP**.

## License
WiPP v 1.0 is release under the [MIT License](LICENSE.md).

## Operating System Compatibility
WiPP has been tested successfully with:
- Ubuntu 16 (Xenial Xerus)

Ubuntu 18 (Bionic Beaver) is not supported yet due to lacking support of incorporated R packages.

## Requirements
- conda ([Bioconda website - Python 3.x](https://conda.io/en/latest/miniconda.html))
- libnetcdf11 ([Ubuntu packages website](https://packages.ubuntu.com/xenial/libs/libnetcdf11))

## Installation
You can install **WiPP** using the following command:
```bash
git clone https://github.com/bihealth/wipp_dev.git
cd wipp_dev
make
```
Now you are ready to run **WiPP**!

## Running a test project

### Change into example_project directory
The pipeline needs to be run from the project directory (the one that contains the `config.yaml` file). Use the following command to change to the example project directory:
```bash
cd ./projects/example_project
```

### Generate training data
From there, you can now run the first part of the pipeline, the generation of the training data.
You can adjust the number of cores using the inline paramter `-n <CORES>`:
```bash
../../run_WiPP.sh tr -n 4 
```
(Note: Running this for the first time takes a while: another conda environment is created)

### Annotate detected peaks
Run the following command to start the annotation:
```bash
../../run_WiPP.sh an
```
The script opens a simple visualization tool using the default pdf viewer, and will wait for you to assign a class to the peak. By default, seven different classes are available for you to choose from.
Once you have annotated the required amount of peaks for each algorithm (only 25 for the example project), the tool will automatically close. \
You are now ready to launch the last part of the pipeline. 

### Do the actual peack picking
Many sequential substep are in fact happening during this final step of the pipeline such as classifier hyperparameter optimisation, peak detection algorithms paramter optimisation, peak detection on the full dataset, output classification and result integration. \
Run it with the following command and adjust the number of cores using the inline paramter `-n <CORES>`:
```bash
../../run_WiPP.sh pp -n 4 
```

## Running your own project
To run your own project, you have to do some prelimitary work and decisions, which is described in this section.
To actually run **WiPP** subsequently, you have to follow the same steps as in the example_project.

### Input files
Let's create and structure the directory for your data files. **WiPP** supports mzML, mzData and NetCDF file formats.

You can create the `Input_folder` directory anywhere you want as long as it is accessible to the tool. You can also name it the way you want, but for clarity purposes, we will call it `Input_folder` in this tutorial. 

Here is the structure:
```
Input_folder/
	condition_1/
	condition_2/
	...
	condition_n/
	Training_samples/
	Optimization_samples/
	Wash/
```

All you files should be separated into subdirectories corresponding to the experimental conditions of your study (here, condition 1 to n).

Three extra subdirectories are necessary to run the pipeline.

The `Wash` directory is optional but if present should contain the wash or blank sample files. Here, we assume that these samples contain the alkanes that are used for Retention Index calculation.

The `Training_samples` directory should contain a subset of the pooled samples or a representative subset of the samples of each biological condition. We recommend using a minimum of 2 sample for each condition.

The `Optimization_samples` directory is similar to the `Training_samples` directory, just make sure to choose different samples as we do not want the parameters to be optimized using the same data the classifier was trained on. 

> ### Note
> If you are wondering if the data files used as training and optimization samples should also be present with the full dataset, the answer is yes! Those files are just present in two copies in different places.
