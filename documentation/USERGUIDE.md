# WiPP - A **W**orkflow for **i**mproved **P**eak **P**icking - User Guide

**WiPP** is an open source large scale GC-MS data preprocessing workflow built in Python 3 that uses machine learning to optimise, automate and combine the peak detection process of commonly used peak picking algorithms.

**WiPP** has been developed as a collaborative effort between the Berlin Institute of Health (BIH) Metabolomics platform, the BIH Core Unit Bioinformatics, the INRA Plateforme d'Exploration du Métabolisme, and the INRA Laboratoire d'Etude des Résidus et Contaminants dans les Aliments.

<a name="pub"></a>
## Publication

Borgsmüller N, Gloaguen Y, Opialla T, Blanc E, Sicard E, Royer A-L, Le Bizec B, Durand S, Migné C, Pétéra M, Pujos-Guillot E, Giacomoni F, 
Guitton Y, Beule D, Kirwan J. WiPP: Workflow for Improved Peak Picking for Gas Chromatography-Mass Spectrometry (GC-MS) Data. Metabolites. 2019; 
9(9):171. https://doi.org/10.3390/metabo9090171

***

## User Guide Overview

* [Target audience](#target-audience)
* [How does WiPP work?](#how-it-works)
* [Tell me more about WiPP](#more-about-wipp)
    * [Workflow overview](#workflow-overview)
        * [1. Training data generation](#overview-tr)
        * [2. Classifier training and parameter optimisation](#overview-an)
        * [3. High confidence peak set generation](#overview-pp)
* [What data types does WiPP support?](#input-data-types)
* [How do I use WiPP?](#how-to-use)
    * [Installation and requirements](#installation)
        * [Requirements](#requirements)
            * [Changing your conda version](#conda-version)
        * [Installing WiPP](#installing-wipp)
    * [Running the pipeline](#running-pipeline)
        * [(Step 1) Training data generation](#running-tr)
        * [(Step 2) Training data annotation](#running-an)
        * [(Step 3) Optimisation and high confidence peak set generation](#running-pp)
        * [Running WiPP on an HPC cluster](#running-hpc)
* [Final output](#final-output)
* [Tool architecture](#tool-architecture)
    * [Input directory](#input-dir)
* [Pipeline settings](#pipeline-settings)
    * [General parameters](#general-params)
    * [Training data parameters](#tr-params)
    * [Optimization parameters](#opt-params)

***
<a name="target-audience"></a>
## Target audience

You do not need to be a bioinformatician or a computer scientist to use **WiPP**. However, you need to be comfortable with a terminal and used to command line tools. For example, if absolute and relative paths do not mean anything to you, we recommend that you seek assistance to a bioinformatician or familiarise yourself with some basics before starting.
If you plan to use this tool on an HPC, you will probably need some help from your HPC admin, but do not worry, we give some examples on how to set it up later in this guide.

Using **WiPP** also requires some basic understanding of GC-MS analyses and the underlying data. If you are not familiar with GC-MS at all, this tool is probably not right for you. But we hope to see you soon when the time comes!

<a name="how-it-works"></a>
## How does **WiPP** work?

**WiPP** uses machine learning approaches to classify the quality of peaks detected in GC-MS data, by optimizing the parameter settings 
for the **XCMS CentWave** and **XCMS MatchedFilter** algorithms. The workflow then classifies and combines the peaks identified by 
individual algorithms to generate a final high-quality peak set, automatically removing false positive peaks.

<a name="more-about-wipp"></a>
## Tell me more about WiPP

**WiPP** is implemented in Python 3 using **[Snakemake](https://snakemake.readthedocs.io/en/stable/)**, a reproducible and scalable workflow management system, and distributed via github under the [MIT licence](../LICENSE.md).

**WiPP** offers a modular design to allow the addition of other existing peak picking algorithm and can be run on a local computer or on a High 
Performance Computing cluster (HPC). It has been tested on Ubuntu 16 (Xenial Xerus), Ubuntu 20 (Focal Fossa) and CentOS Linux 7 (Core) operating 
systems.

<a name="workflow-overview"></a>
### Workflow overview

There are 3 main steps to the workflow that you will need to follow to go from your raw data to a high confidence peak set.

<a name="overview-tr"></a>
#### 1. Training data generation

The first step consists in generating a training data set that the chosen classifier can learn from. The training data is generated using a subset of the dataset to be analysed, it can either be pooled sample files or a subset of the sample files from each of the biological conditions present in your study; we recommend a minimum of 2 sample files from each condition. It is important that all biological conditions are represented in the training data for optimal performance. 

The chosen peak picking algorithms are then run on this data with a wide range of parameter combinations to generate a list of detected peaks for each representative sample. Those peaks need to be manually assessed and classify into different categories (True positive peaks, Noise etc...) using an embedded peak visualisation and classification tool. This is the only manual step that you will have to perform!

Once the peaks are classified, your training data is ready and we can move on to the next step and train the model.

<a name="overview-an"></a>
#### 2. Classifier training and parameter optimisation

There are in fact two underlying parts to this step, but for ease of use and clarity purposes, those are kept hidden to the user.

The first part consists of training and validating the classifier, 20% of the training data is kept aside for validation and overfitting check purposes, the remaining 80% is used for hyperparameter optimization and training. The accuracy of the resulting classifier is measured using the validation set.

The second part performs the actual optimization of peak detection algorithms parameters and relies on a scoring function to assess the performance of the different parameter sets. To do so, the algorithms are run on a subset of the dataset using, again, a wide range of parameter sets for each algorithm. The best performing parameters are then selected based on the calculated scores to maximize the quality and quantity of detected peaks.

<a name="overview-pp"></a>
#### 3. High confidence peak set generation

At this stage, the tool has established the optimal parameters of the selected peak picking algorithms for the data being analysed. These algorithms are now run onto the entire dataset and the detected peaks are classified using our trained classifier. The outputs from the different algorithms are then merged and true positive peaks reported as high confidence peak set.

That is all the basics you need to know about **WiPP**! Now let's get started with the technical part on how to use the tool. 

*To learn more about the underlying data processing such as the scoring function, baseline correction or optimization, please see the [WiPP 
publication](http://dx.doi.org/10.3390/metabo9090171).*

<a name="input-data-types"></a>
## What data types does WiPP support?

**WiPP** can be used to analyse high- or low-resolution GC-MS data acquired using an Electron Impact ionization source and best performs with a 
minimum acquisition rate of 10 scans/s.

The supported file formats for input data are mzML, mzData and NetCDF.

> ### Note
> If you want to analyse data that is not in one of the supported formats, you can use [Msconvert](http://proteowizard.sourceforge.net/index.html) from ProteoWizard which supports the conversion of most MS proprietary file formats.

<a name="how-to-use"></a>
## How do I use WiPP?

Now that you have an overview of what **WiPP** really is, let's learn how to install and use it.

<a name="installation"></a>
### Installation and requirements

*[Skip to the next section](#running-pipeline) if you have already installed **WiPP** on your computer or HPC.*

<a name="requirements"></a>
#### Requirements
**WiPP** uses libnetcdf11 and Bioconda. To install these tools, please follow the installation instructions on the external links below:
* [libnetcdf11](https://packages.ubuntu.com/xenial/libs/libnetcdf11) (Ubuntu packages website)
* [Bioconda](https://conda.io/en/latest/miniconda.html) (aka conda, miniconda)

> ### Note
> If you have freshly installed conda, remember to run `source ~/.bashrc` or open a new terminal before installing **WiPP**.

<a name="conda-version"></a>
##### Changing your conda version
**WiPP** requires conda to be between version 4.3 and 4.9 (inclusive) -- conda version 4.9.2 is preferred.
Conda versions < 4.4 are not supported for external analysis (i.e. running on an HPC cluster)

You can check the version of conda you have currently installed, by running `conda --version`  
e.g.

```
+-> conda --version
conda 4.10.1
```

To change your conda to version 4.9.2, run `conda install conda=4.9.2` and press `enter` (or `y`) when asked if you want to proceed to install 
and upgrade/downgrade the listed packages.

<a name="installing-wipp"></a>
#### Installing WiPP

Once Bioconda (`conda`) is enabled, you can install **WiPP** using this command:

```
git clone https://github.com/bihealth/WiPP.git
cd WiPP
make
```

You are now all set up and ready to run **WiPP**!

<a name="running-pipeline"></a>
### Running the pipeline

In this section, we will see how to run the pipeline. Once everything is in place, only a few commands are required to run **WiPP** and 
generate our high confidence peak set from the raw data.

We recommend using `projects/example_project/` to test **WiPP** before jumping to the analysis of your own data.

If you prefer to start right away with your own data, please refer to the [Input directory](#input-dir) section in this document to learn how to 
organize the directories containing your raw data files.

You may also wish to skip ahead and learn more about the [pipeline settings](#pipeline-settings) before continuing with the instructions below on 
how to run **WiPP**.


<a name="running-tr"></a>
#### Training data generation (Step 1)

The pipeline needs to be run from a subdirectory of `WiPP/projects/` (i.e. the directory that contains the `config.yaml` file). 

Use the following command to set `example_project` as your current directory:

```
cd WiPP/projects/example_project/
```
From there, you can now run the first part of the pipeline, the generation of the training data. You can adjust the number of cores using the inline parameter -n <CORES>:

```
../../run_WiPP.sh tr -n 4
```

This step can take some time depending on the number of cores you have allocated for the job. Once the job has finished, you can move on to the manual step to generate the training data.

<a name="running-an"></a>
#### Training data annotation (Step 2)

You now need to annotate manually a series of peaks which will then be used to train the SVM classifiers and subsequently optimise the peak picking algoritm parameters to best fit your data. This task usually take several hours, but you only need to do that once per instrument/protocol. Once trained, the SVM classifiers can be reused on other datasets generated with the same instrument and data acquisition method.

To start the classification run the following command, you can add the flag `--help` to learn more:

```
../../run_WiPP.sh an
```

The script opens a simple visualization tool using the default pdf viewer, and will wait for to assign a class to the peak. Seven different classes are available for you to choose from.
Once you have annotated the required amount of peaks for each algorithm (1200 by default), the tool will automatically close. You are now ready to launch the last part of the pipeline.

> ### Note
> The default parameters are only taken if they are not defined in the `config.yaml` file. Chances are that you used example config file that we provided as a template, make sure to adapt the parameters properly. For instance, the number of peaks to annotated is set to 25 per sample in the example file, make sure to change this parameter before running this step.

<a name="running-pp"></a>
#### Optimisation and high confidence peak set generation (Step 3)

Many sequential substep are in fact happening during this final step of the pipeline such as classifier hyperparameter optimisation, peak detection algorithms parameter optimisation, peak detection on the full dataset, output classification and result integration.
Run it with the following command and adjust the number of cores using the inline parameter -n <CORES>:

```
../../run_WiPP.sh pp -n 4 
```

<a name="running-hpc"></a>
#### Running WiPP on an HPC cluster

This section is meant to provide additional information for advanced users who wish to set up **WiPP** on a computing cluster for routine use. 

Running **WiPP** on a cluster requires the user to be logged into an HPC node with access to a SLURM scheduler. (i.e. jobs are submitted using 
the `sbatch` command)

> ### Note
> The **WiPP** annotation step (`an`) is interactive and thus cannot be run externally on HPC.

To submit a **WiPP** training (`tr`) or peak picking (`pp`) job to the HPC scheduler, include the `-x` flag (plus optional parameters):  
e.g.
```bash
../../run_WiPP.sh tr -x
```
```bash
../../run_WiPP.sh pp -x -g 10 -n 7
```

HPC job parameter details:
  * **-x, --external** : Submit this job to the cluster scheduler *(REQUIRED)*
  * **-g <GIGS>, --gigs-per-cpu <GIGS>** : Value is passed to the scheduler via the `sbatch` parameter `--mem-per-cpu <GIGS>G` *(OPTIONAL: default 6)*
  * **-n <NODES>, --nodes <NODES>** : Value is passed to the scheduler via the `sbatch` parameter `--cpus-per-task <NODES>` *(OPTIONAL: default 4)*  

When a **WiPP** job is run on the cluster, a `slurm_log/` subfolder is created to store the 
`wipp_<pipeline-step>_<project-name>_<cluster-job-number>.log` files containing the job's sys-out messages.  
e.g.  
`projects/my_project/slurm_log/wipp_pp_installation_test-3089255.log`

> ### Note
> *For information on how to monitor cluster jobs or troubleshoot scheduling errors, please refer to your local HPC user documentation.*

<a name="final-output"></a>
## Final output

All result files are created in a new subfolder named `04_Final_results/`. 

Two types of results are available, first a `*.csv` file (wide format) contains the feature table, every row represent a peak and each column a 
specific information such as the retention time, retention index or the spectrum (peak mz:peak area) for a specific sample. Missing peaks are 
represented by missing values in the corresponding sample column.

The second output is a `*.msp` file which contains the spectra of every peak reported in the `*.csv` file and can be used for identification 
purposes (Using a dedicated GC-MS peak identification tool).

A `*.csv/*.msp` pair is available for every biological condition as well as all samples together.

<a name="tool-architecture"></a>
## Tool architecture

Once you have your GC-MS data ready in [one of the three supported formats](#input-data-types), and **WiPP** is [installed](#installation), let's 
see how the WiPP **subdirectories** are organised:

```
WiPP/
├── bin/
├── documentation/
├── envs/
├── lib/
│   ├── python/
│   └──  R/
├── pp_configs/
└── projects/
    ├── example_project/
    └── installation_test/
```

The `bin` folder contains the snakemake files, which describe the entire workflow and how it should be run. You do not need to touch this unless you want to modify the workflow or implement new steps. This folder also contains an optional file to define the cluster settings, we will tell you more about that later.

The `pp_configs` directory contains one setting file for each peak detection algorithm used in the pipeline. Yes, the entire point of this pipeline is to optimise those parameters, but only a subset of the parameters are actually optimised. We recommend not to edit those files unless you are very familiar with the peak detection algorithms and their settings.

The `lib` directory is the core of **WiPP**, it contains python and R code used to run the pipeline. Again, no need to look into this directory unless you are a developer.

The last and most important directory for end users is called `projects`. Each time you run the pipeline, the output will be stored in this folder, and every new analysis needs a new subfolder. As an example, we created the `example_project` directory which contains an example config file for this project. When starting a new analysis, you will have to create a new directory with a new config file saved under the name `config.yaml`. We will go through the settings together in the next section.

At this stage, you probably noticed that we are missing the data input directory. Do not panic, this is totally normal. The input folder that contains all your files can be anywhere on the system as long as it is accessible by the tool, and its location will be defined in the pipeline settings.

<a name="input-dir"></a>
### Input directory

Here is an overview of the input directory and how it should be structured.

> ### Note
> You can name your input folder the way you want, for clarity purposes we will just call it `input_folder` here. 

```
input_folder/
├── condition_1/
├── condition_2/
├── ...
├── condition_n/
├── training_samples/
├── optimization_samples/
└── wash/
```

All the files of your dataset should be structured into subdirectories corresponding to the biological conditions of your study.
You may follow the above the example, and simply call them `condition_1/` to `condition_n`, or use more descriptive names.

The `wash/` directory is optional -- if present should contain the wash or blank sample files. 
Here we assume that these samples contain the alkanes that are used for Retention Index calculation. 
If you do not have blank samples containing alkanes, you can omit this folder (as is the case in our `example_project`).

The `training_samples` directory should contain the subset of your data to be used for the training of the classifier.
It should contain a subset of the pooled samples or a representative subset of the samples of each biological condition. 
We recommend using a minimum of 2 sample for each condition.

The `optimization_samples` directory should contain the subset of your data be used for parameter optimization.
Samples should be selected in a manner similar to the `training_samples` directory, but in order for the algorithms to effectively learn your 
data set **it is important to choose different samples for training & optimization**

> ### Note
> The sample files used for training and optimization should still be present in your sample directories, the files in the `training_samples` and 
`optimization_samples` directories are only copies.

<a name="pipeline-settings"></a>
## Pipeline settings

All general pipeline settings are stored in the `config.yaml` file located in `projects/` subfolder you are analysing. 
This config file needs to be created for every new project. 
e.g. for the `example_project`, you will find the file here:

```
WiPP/
└── projects/
    └── example_project/
        └── config.yaml
```

The `config.yaml` contains three parameter sections:
* the first one relates to the general pipeline parameters, 
* the second section focuses on data training step of the pipeline, 
* and the last section concerns the optimization part of the pipeline

<a name="general-params"></a>
### General parameters

The first block of parameters looks like this:
"""
static_data:
  absolute_path:
  high_resolution:
  cores:
  peak_min_width:
  peak_min_mz:
  merging_RT_tol:
"""

The `absolute_path` parameter should point to your input data folder where your raw data, training data and optimization data is stored (and 
potentially your wash sample files). This is the folder we named `input_data` in the previous section.

`high_resolution` should be `True` or `False` depending on your data type.

`cores` defines the number of cores to use. If no value is given, the pipeline will use the maximum number of cores available on your computer.
This value can also be set as an inline parameter when running the workflow.

`peak_min_width` represents the minimum width of the peaks in second to be detected. The default value is `0.5`.

`peak_min_mz` represents the minimum number of m/z to be detected within the same time window to be considered a peak. The default value is `3`.

***

The `algorithms:` block gives you the opportunity to define the peak picking algorithms you want to use. 
Currently, two algorithms are available (XCMS CentWave and MatchedFilter), and you can choose to use either or both of them by setting them to `True` or `False`.

***

The `default_configs:` parameters block allows you to set the path where the default algorithm parameters are stored. Unless you are making your 
own changes to the tool, you can leave this section as it is.

***

Finally, parameter block 
"""
merging:
  RT_tol:
"""
is the time tolerance given in second for peaks detected in the same sample with different parameters/algorithms to be considered the same and reported as one. The default value is `0.2`.

***

<a name="tr-params"></a>
### Training data parameters

The `training_data-general` block allows you to define the location of your `training_samples` as a relative path from your `input_folder` 
directory.
The second parameter is the number of `plots_per_samples` to be generated, these plots will then be used for manual annotation and generation of 
the training dataset. The default value is `200`.

Within the `training_data-params` parameter block, you can specify the algorithm parameters you would like to use to generate your training data 
by defining a list for every entry. The possible combinations will be automatically evaluated by **WiPP** and used to generate the training dataset.
The default values give a wide range of parameter sets and should fit most data.

> ### Note
> The more parameters you set, the longer it will take to run! Consult your MS expert to make an informed decision on the best parameter range to run.

***

<a name="opt-params"></a>
### Optimization parameters

In the `classifier` block, set `SVM` and `RF` (support vector machines and Random forests) to `True` or `False` depending on the model you want 
to use. If both are set to `True`, only the best performing estimator will be conserved. 
`SVM` have shown better performance on all datasets tested so far. 
The `validation_set_size` and `cross_validation_splits` parameters, respectively set to `0.2` and `3` by default are used for hyperparameter 
optimization and to prevent overfitting. We recommend using the default values if you are not familiar with cross-validation procedure to 
evaluate estimator performance.

The `optimization-general` block allows you to define the location of your `optimization_samples ` as a relative path from your `input_folder` 
directory.

The `optimization-score` is a list of parameters used to find the optimal peak picking algorithm parameter set. Default parameters have shown to 
give appropriate results for both high and low-resolution data, but you can change them according to your needs.

Finally, the last settings, `grid_search-params` concern the grid search range for the parameter optimization. Same as the training data, you can 
limit the range if you already familiar with the data produced by your GCMS experiment.

***

The following parameter block is dedicated to retention index, if you do not use alkanes, you can remove or skip this section altogether as the 
default value is `False`.

"""
retention_index:
  wash:
  alkanes:
"""

The `wash:` parameter requires the relative path from the `input_folder` to the `wash_folder`. In our example, this parameter would look like 
this `./wash`

The `alkanes` parameter is the list of alkanes that should be used for retention index calculation. 
The parameter should be a comma separated list of the number of carbons for each alkane.
e.g.  
`"c10","c12","c16",[...],"c32","c34"`

***

Finally, the `alignment:` block allows you to define the parameters for cross sample alignment. The parameter `RI_tol` or `RT_tol` 
will be used as appropriate, depending on your project. 

The value of`RT_tol` should be defined in seconds. 

The `min_similarity:` parameter corresponds to the minimum spectra similarity for peaks to be aligned, and the `min_samples:` parameter defines 
the minimum number of samples (as a ratio) to be present in order for a peak to be reported.
