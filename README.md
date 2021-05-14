# WiPP - A **W**orkflow for **i**mproved **P**eak **P**icking - Quick Start Guide
**WiPP** is an open source large scale GC-MS data preprocessing workflow built in Python 3 that uses machine learning to optimise, automate and combine the peak detection process of commonly used peak picking algorithms.

**WiPP** has been developed as a collaborative effort between the Berlin Institute of Health (BIH) Metabolomics platform, the BIH Core Unit Bioinformatics, the INRA Plateforme d'Exploration du Métabolisme, and the INRA Laboratoire d'Etude des Résidus et Contaminants dans les Aliments.

This document aims to help you get started with **WiPP** and brings you through the minimum requirements to install, set up, and run the software. However, we strongly recommend you to read through the [complete user guide](documentation/USERGUIDE.md) for a full and advanced use of **WiPP**.

## License
**WiPP** v 1.0 is release under the [MIT License](LICENSE.md).

## Operating System Compatibility
**WiPP** has been tested successfully on:
- CentOS 7
- Ubuntu 20
- Ubuntu 16

## Requirements
- conda version >= 4.3.34 ([Bioconda website - Python 3.x](https://conda.io/en/latest/miniconda.html))
- libnetcdf11 ([Ubuntu packages website](https://packages.ubuntu.com/xenial/libs/libnetcdf11))

## Installation

### Installing miniconda
*(This section can be skipped if you already have conda installed)*

In a linux terminal, get the python version currently in use on your system. e.g.  
```
+-> python --version
Python 3.7.1
```

In a browser, open: `https://docs.conda.io/en/latest/miniconda.html#linux-installers`  
and right-click on the "Miniconda3 Linux 64-bit" link correspondig to your python version, and select "Copy link address".

In your linux terminal, go to your tmp directory, type `wget ` and paste the miniconda URL from your clipboard, and then hit enter to initiate 
downloading the miniconda installer. e.g.

```
+-> cd ~/scratch/tmp
+-> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86.sh
--2021-05-06 17:27:49--  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86.sh
Resolving repo.anaconda.com (repo.anaconda.com)... 104.16.130.3, 104.16.131.3, 2606:4700::6810:8303, ...
Connecting to repo.anaconda.com (repo.anaconda.com)|104.16.130.3|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 65741329 (63M) [application/x-sh]
Saving to: ‘Miniconda3-latest-Linux-x86.sh’

100%[========================================================================================>] 65,741,329  60.3MB/s   in 1.0s

2021-05-06 17:27:50 (60.3 MB/s) - ‘Miniconda3-latest-Linux-x86.sh’ saved [65741329/65741329]
+-> 
```

You now need to create a new text file, we'll call it `miniconda_hash.txt`, where:
* the first value is the `SHA256 hash` copied from the minconda installers page in your browser
* the separator is **2 spaces**
* the second value is the name of the `*.sh`miniconda installer file you just downloaded

e.g. `f387eded3fa4ddc3104b7775e62d59065b30205c2758a8b86b4c27144adafcc4  Miniconda3-latest-Linux-x86.sh`

Now run `sha256sum` using the above file to confirm your download was successful. e.g.  
```
+-> sha256sum -c miniconda_hash.txt
Miniconda3-latest-Linux-x86.sh: OK
```

Now initiate the miniconda installation by running the installer, e.g. 
```
+-> bash Miniconda3-latest-Linux-x86.sh
```
and follow the prompts on the installer screens.

(note: If you are unsure about any setting, accept the defaults. You can change them later.)

Finally, you must initialize miniconda in your terminal by running:
```
+-> source ~/.bashrc
```

To test your miniconda installation, run the command `conda list` in your terminal.
A list of installed packages will appear if it has been installed correctly.

### Installing WiPP

You can install **WiPP** using the following command:
```bash
git clone https://github.com/bihealth/WiPP.git
cd WiPP
make
```
Now you are ready to run **WiPP**!

## Running the WiPP installation test

After downloading & installing WiPP, you can use the installation test project to confirm your installation of WiPP runs as expected.

First, go to the project directory:
`cd WiPP/projects/installation_test`

Now run WiPP peak picking using the following command, adjusting the number of cores using the inline parameter -n as appropriate.

`../../run_WiPP.sh pp -n 1`

Note: Using a single core, the run should complete in slightly over 1 hour and use 6G of memory.

When the peak picking run is done, first check whether the system out logging messages end with the following lines, confirming the job 
successfully completed all steps:
```
Finished job 0.
48 of 48 steps (100%) done
Complete log: <$path to snakemake log file>
```

And finally, run the following commands to identify any differences between your output and the expected results:
```
diff 04_Final_results/all__final.csv expected_output/all__final.csv
diff 04_Final_results/all__final.msp expected_output/all__final.msp
diff 04_Final_results/Liver__final.csv expected_output/Liver__final.csv
diff 04_Final_results/Liver__final.msp expected_output/Liver__final.msp
```

If your installation ran as expected, all 4 `diff` commands should return nothing (0 lines).

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
> ### Note
> Running this for the first time takes a while: another conda environment is created

### Annotate detected peaks
Run the following command to start the annotation:
```bash
../../run_WiPP.sh an
```
The script opens a simple visualization tool using the default pdf viewer, and will wait for you to assign a class to the peak. By default, seven different classes are available for you to choose from.
Once you have annotated the required amount of peaks for each algorithm (only 25 for the example project), the tool will automatically close.
You are now ready to launch the last part of the pipeline.

### Do the actual peack picking
Many sequential substeps are in fact happening during this final step of the pipeline such as classifier hyperparameter optimisation, peak detection algorithms parameter optimisation, peak detection on the full dataset, output classification and result integration.
Run it with the following command and adjust the number of cores using the inline parameter `-n <CORES>`:
```bash
../../run_WiPP.sh pp -n 4
```

> ### Note
> As you annotated a very small amount of peaks, your classifier is likely not to be accurate. For this reason, we provide a trained classifier. In order to use this example classifier, please uncomment the line `path: example_classifier` (by removing the `#` symbol) in the `classifier` block of the `config.yaml` file. You can follow the same procedure for your data if you want to use a specific classifier for several projects. Please note that the example classifier is provided to help you test the tool and is specific to the test data, do not use it for your own project.


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

All your files should be separated into subdirectories corresponding to the experimental conditions of your study (here, condition 1 to n).

Three extra subdirectories are necessary to run the pipeline.

The `Wash` directory is optional but if present should contain the wash or blank sample files. Here, we assume that these samples contain the alkanes that are used for Retention Index calculation.

The `Training_samples` directory should contain a subset of the pooled samples or a representative subset of the samples of each biological condition. If you do not have pooled samples, we recommend using a minimum of 2 sample for each condition.

The `Optimization_samples` directory is similar to the `Training_samples` directory, just make sure to choose different samples as we do not want the parameters to be optimized using the same data the classifier was trained on.

> ### Note
> The sample files used for training and optimization should still be present in your sample directories, the files in the `Training_samples` and `Optimization_samples` directories are only copies.

### Pipeline settings

This tutorial only shows the minimum requirements to run the pipeline, to learn more about all pipeline settings, have a look at the pipeline settings section of the [complete user guide](documentation/USERGUIDE.md).

All general pipeline settings are stored in the `config.yaml` of the individual project folder and need to be created for every new project. You can have a look at the [example config](projects/example_project/config.yaml) from the example_project.

> ### Note
> By default, in the example project, we run only one peak picking algorithm to speed up the test. Do not forget to enable in the `config.yaml` file the peak picking algorithms you want to use.

First, you need to define the absolute path to your `Input_folder`. This parameter can be found in the `static_data` block under the name `absolute_path`. You also need to specify the resolution of your data by setting the `high_resolution` parameter to `True` or `False` (in this context, any data with a mass resolution higher than 1 Da is considered high resolution).

Next, if you have a `Wash` directory for retention index calculation, go to the `retention_index` block. You need to define the relative path from the `Input_folder` to the `Wash` folder. In our example, this parameter would look like this `./Wash`.
You also have to define the alkanes present in your samples as a simple list `c10,c12,c16,[...],c32,c34`.
If you do not have blank samples containing alkanes for retention index calculation, you can leave this section as it is in the example file.

The last two compulsory parameters that you need to define are the relatives path from the `Input_folder` to your `Training_samples` and `Optimization_samples` directories, respectively found in `training_data-general` and `optimization-general` setting blocks.

That's all for the basic settings, you are now ready to run the pipeline.

> ### Note
> Keep in mind that those parameters are the only one required to run the pipeline, but there is a lot more you can do to precisely tune the pipeline. Have a read through the [complete user guide](documentation/USERGUIDE.md) to learn more.
 
### Running the pipeline
Follow the same steps as in the [example_project](#running-a-test-project).

> ### Note
> Peak annotation usually takes several hours (1200 peaks per algorithm by default), but you only need to do that once per instrument/protocol. Once trained, the SVM classifiers can be reused on other datasets generated with the same instrument and data acquisition method.

### Results

All result files are created in a new subfolder named 04_Final_results. 

Two types of results are available, first a csv file (wide format) contains the feature table, every row represent a peak and each column a specific information such as the retention time, retention index or the spectrum (peak mz:peak area) for a specific sample. Missing peaks are represented by missing values in the corresponding sample column.

The second output is a .msp file which contains the spectra of every peak reported in the .csv file and can be used for identification purposes (Using a dedicated GC-MS peak identification tool).

A .csv/.msp pair is available for every biological condition as well as all samples together.
