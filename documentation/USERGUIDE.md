# WiPP - A **W**orkflow for **i**mproved **P**eak **P**icking - User Guide

**WiPP** is an open source large scale GC-MS data preprocessing workflow built in Python 3 that uses machine learning to optimise, automate and combine the peak detection process of commonly used peak picking algorithms.

**WiPP** has been developed as a collaborative effort between the Berlin Institute of Health (BIH) Metabolomics platform, the BIH Core Unit Bioinformatics, the INRA Plateforme d'Exploration du Métabolisme, and the INRA Laboratoire d'Etude des Résidus et Contaminants dans les Aliments.

## Target audience

You do not need to be a bioinformatician or a computer scientist to use **WiPP**. However, you need to be comfortable with a terminal and used to command line tools. For example, if absolute and relative paths do not mean anything to you, we recommend that you seek assistance to a bioinformatician or familiarise yourself with some basics before starting.
If you plan to use this tool on an HPC, you will probably need some help from your HPC admin, but do not worry, we give some examples on how to set it up later in this guide.

Using **WiPP** also requires some basic understanding of GC-MS analyses and the underlying data. If you are not familiar with GC-MS at all, this tool is probably not right for you. But we hope to see you soon when the time comes!

## How does **WiPP** work?

The tool uses machine learning approaches to classify the quality of the peaks detected by existing methods (currently XCMS CentWave & XCMS MatchedFilter) to optimise the parameter settings of each individual algorithms. The workflow then classifies and combine peaks identified by individual algorithms to generate a final high-quality peak set, automatically removing false positive peaks.

## Tell me more about WiPP

**WiPP** is implemented in Python 3 using **[Snakemake](https://snakemake.readthedocs.io/en/stable/)**, a reproducible and scalable workflow management system, and distributed via github under the [MIT licence](../LICENSE.md).

**WiPP** offers a modular design to allow the addition of other existing peak picking algorithm and can be run on a local computer as well as on an HPC. It has been tested on Ubuntu 16 (Xenial Xerus) and CentOS 7.6.1810 (Core) operating systems.

### Workflow overview

There are 3 main steps to the workflow that you will need to follow to go from your raw data to a high confidence peak set.

#### 1. Training data generation

The first step consists in generating a training data set that the chosen classifier can learn from. The training data is generated using a subset of the dataset to be analysed, it can either be pooled sample files or a subset of the sample files from each of the biological conditions present in your study; we recommend a minimum of 2 sample files from each condition. It is important that all biological conditions are represented in the training data for optimal performance. 

The chosen peak picking algorithms are then run on this data with a wide range of parameter combinations to generate a list of detected peaks for each representative sample. Those peaks need to be manually assessed and classify into different categories (True positive peaks, Noise etc...) using an embedded peak visualisation and classification tool. This is the only manual step that you will have to perform!

Once the peaks are classified, your training data is ready and we can move on to the next step and train the model.

#### 2. Classifier training and parameter optimisation

There are in fact two underlying steps to this part, but for ease of use and clarity purposes, those are kept hidden to the user.

The first part consists of training and validating the classifier, 20% of the training data is kept aside for validation and overfitting check purposes, the remaining 80% is used for hyperparameter optimization and training. The accuracy of the resulting classifier is measured using the validation set.

The second part performs the actual optimization of peak detection algorithms parameters and relies on a scoring function to assess the performance of the different parameter sets. To do so, the algorithms are run on a subset of the dataset using, again, a wide range of parameter sets for each algorithm. The best performing parameters are then selected based on the calculated scores to maximize the quality and quantity of detected peaks.

#### 3. High confidence peak set generation

At this stage, the tool has established the optimal parameters of the selected peak picking algorithms for the data being analysed. These algorithms are now run onto the entire dataset and the detected peaks are classified using our trained classifier. The outputs from the different algorithms are then merged and true positive peaks reported as high confidence peak set.

That is all the basics you need to know about **WiPP**! Now let's get started with the technical part on how to use the tool. 

> ### Note
> If you want to know more about the underlying data processing such as the scoring function, baseline correction or optimization, take a look at the [paper available]() (empty link until paper published).


## What data do WiPP supports?

**WiPP** can be used to analyse high or low-resolution GC-MS data acquired using an Electron impact ionization source and best performs with a minimum acquisition rate of 10 scans/s.

The supported file formats are mzML, mzData and NetCDF.

## How do I use WiPP?

Now that you have an overview of what **WiPP** really is, let's learn how to use it.

### Prerequisite

If you want to test the software before jumping to the analysis of your own data, which we recommend, you can use the example dataset. If you prefer to start right away with your data, you can skip to the [install section](#Install).

If you want to use your own data but do not have it in one of the supported formats, you can use [Msconvert](http://proteowizard.sourceforge.net/index.html) from ProteoWizard which supports the conversion of most MS proprietary file formats.

### <a name="Install"></a> Install

#### Dependencies

Have you already install **WiPP** on your computer or HPC? Go to the [next section](#Pipeline-architecture) to learn about the architecture of the tool.

**WiPP** uses Bioconda. So first, you will need to install conda following the instruction on the **[Bioconda website](https://conda.io/en/latest/miniconda.html)**.

You also need to install libnetcdf11 ([Ubuntu packages website](https://packages.ubuntu.com/xenial/libs/libnetcdf11))

> ### Note
> If you install conda from scratch, remember to `source ~/.bashrc` or open a new terminal before installing WiPP.

#### Installation

Once Bioconda is enabled, you can install **WiPP** using this command:

```
git clone https://github.com/bihealth/WiPP.git
cd WiPP
make
```

You are all set up and ready to run, let us take a look at the architecture of the tool to know where to put the data and define the pipeline settings.


### <a name="Pipeline-architecture"></a> Tool architecture

Now that you have your data ready in either one of the 3 supported format, and **WiPP** installed, let see how the tool is organised:

```
WiPP/
	bin/
	documentation/
	envs/
	lib/
		python/
		R/
	pp_configs/
	projects/
		example_project/
```

The `bin` folder contains the snakemake files, which describe the entire workflow and how it should be run. You do not need to touch this unless you want to modify the workflow or implement new steps. This folder also contains an optional file to define the cluster settings, we will tell you more about that later.

The `pp_configs` directory contains one setting file for each peak detection algorithm used in the pipeline. Yes, the entire point of this pipeline is to optimise those parameters, but only a subset of the parameters are actually optimised. We recommend not to edit those files unless you are very familiar with the peak detection algorithms and their settings.

The `lib` directory is the core of **WiPP**, it contains python and R code used to run the pipeline. Again, no need to look into this directory unless you are a developer.

The last and most important directory for end users is called `projects`. Each time you run the pipeline, the output will be stored in this folder, and every new analysis needs a new subfolder. As an example, we created the `example_project` directory which contains an example config file for this project. When starting a new analysis, you will have to create a new directory with a new config file saved under the name `config.yaml`. We will go through the settings together in the next section.

At this stage, you probably noticed that we are missing the data input directory. Do not panic, this is totally normal. The input folder that contains all your files can be anywhere on the system as long as it is accessible by the tool, and its location will be defined in the pipeline settings.

Here is an overview of the input directory and how it should be structured:

> ### Note
> You can name your input folder the way you want, for clarity purposes we will just call it `Input_folder` here. 

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

All the files of your dataset should be structure into subdirectory corresponding to the biological conditions of your study (here, condition 1 to n).

The `Wash` directory is optional but if present should contain the wash or blank sample files. Here we assume that these samples contain the alkanes that are used for Retention Index calculation. If you do not have blank samples containing alkanes,  you can ignore this folder as it is the case in our `example_project`.

The other 2 directories should contain a subset of the data which will be used for the  training of the classifier and parameter optimization.

The `Training_samples` directory should contain a subset of the pooled samples or a representative subset of the samples of each biological condition. We recommend using a minimum of 2 sample for each condition.

The `Optimization_samples` directory is similar to the `Training_samples` directory, just make sure to choose different samples as we do not want the parameters to be optimized using the same data the classifier was trained on.

> ### Note
> The sample files used for training and optimization should still be present in your sample directories, the files in the `Training_samples` and `Optimization_samples` directories are only copies.

### Pipeline settings

All general pipeline settings are stored in the `config.yaml` of the individual project folder and need to be created for every new project. You can find an example in the `example_project` directory that you will find here:

```
WiPP/
	projects/
		example_project/
			config.yaml
```

There are three parameter sections, the first one relates to the general pipeline parameters, the second section focuses on data training step of the pipeline, and the last section concerns the optimization part of the pipeline.

#### General parameters

The first block of parameters look like this:

* static\_data
	* absolute\_path
	* high\_resolution
	* cores
	* peak\_min\_width
	* peak\_min\_mz
	* merging\_RT\_tol

The `absolute_path` parameter should point to your input data folder where your raw data, training data and optimization data is stored (and potentially your wash sample files). This is the folder we named `Input_data` in the previous section.

`high_resolution` should be `True` or `False` depending on your data type.

`cores` defines the number of cores to use. If no value is given, the pipeline will use the maximum number of cores available on your computer. This value can also be set as an inline parameter when running the workflow.

`peak_min_width` represents the minimum width of the peaks in second to be detected. The default value is `0.5`.

`peak_min_mz` represents the minimum number of m/z to be detected within the same time window to be considered a peak. The default value is `3`.

***

The `algorithms` block gives you the opportunity to define the peak picking algorithms you want to use. Currently, two algorithms are available (XCMS CentWave and MatchedFilter), and you can choose to use either or both of them by setting them to `True` or `False`.

***

The `default_config` parameters block allows you to set the path where the default algorithm parameters are stored. Unless you are making your own changes to the tool, you can leave this section as it is.

***

Finally, the `RT_tol` parameter in the `merging` parameter block is the time tolerance given in second for peaks detected in the same sample with different parameters/algorithms to be considered the same and reported as one. The default value is `0.2`.

***

#### Training data parameters

The `training_data-general` block allows you to define the location of your `training_samples` as a relative path from your `Input_folder` directory.
The second parameter is the number of `plots_per_samples` to be generated, these plots will then be used for manual annotation and generation of the training dataset. The default value is `200`.

Within the `training_data-params` parameter block, you can specify the algorithm parameters you would like to use to generate your training data by defining a list for every entry. The possible combinations will be automatically evaluated by **WiPP** and used to generate the training dataset. The default values give a wide range of parameter sets and should fit most data.

> ### Note
> The more parameters you set, the longer it will take to run! Consult your MS expert to make an informed decision on the best parameter range to run.

***

#### Optimization parameters

In the `classifier` block, set `SVM` and `RF` (support vector machines and Random forests) to `True` or `False` depending on the model you want to use. If both are set to `True`, only the best performing estimator will be conserved. `SVM` have shown better performance on all datasets tested so far. The `validation_set_size` and `cross_validation_splits` parameters, respectively set to `0.2` and `3` by default are used for hyperparameter optimization and to prevent overfitting. We recommend using the default values if you are not familiar with cross-validation procedure to evaluate estimator performance.

The `optimization-general` block allows you to define the location of your `optimization_samples ` as a relative path from your `Input_folder` directory.

The `optimization-score` is a list of parameters used to find the optimal peak picking algorithm parameter set. Default parameters have shown to give appropriate results for both high and low-resolution data, but you can change them according to your needs.

Finally, the last settings, `grid_search-params` concern the grid search range for the parameter optimization. Same as the training data, you can limit the range if you already familiar with the data produced by your GCMS experiment.

***

The following parameter block is dedicated to retention index, if you do not use alkanes, you can remove or skip this section altogether as the default value is `False`.

* retention_index
	* wash
	* alkanes

The `wash` parameter requires the relative path from the `Input_folder` to the `wash_folder`. In our example, this parameter would look like this `./Wash`

The `alkanes` parameter is the list of alkanes that should be used for retention index calculation. The parameter should be a comma separated list of the number of carbons for each alkanes as follow `"c10","c12","c16",[...],"c32","c34"`.

***

Finally, the `alignment` block allows you to define the parameters for cross sample alignment. The appropriate parameter of`RI_tol` or `RT_tol` will be used depending on your project. `RT_tol` should be defined in seconds. The `min_similarity` parameter corresponds to the minimum spectra simialrity for peaks to be aligned, and the `min_samples` parameter defines the minimum number of samples (as a ratio) for a peak to be present in order to be reported.

## Running the pipeline

We are now at the final stage of this user guide, and in this section, we will see how to run the pipeline. Once everything is in place, only a few commands are required to run **WiPP** and generate our high confidence peak set from the raw data.

### Training data generation (Step 1)

The pipeline needs to be run from the project directory that you created in the previous step (the one that contains the `config.yaml` file). Use the following command to set the location to your current directory, don't forget to change the project name to your own:

```
cd WiPP/projects/example_project/
```
From there, you can now run the first part of the pipeline, the generation of the training data. You can adjust the number of cores using the inline parameter -n <CORES>:

```
../../run_WiPP.sh tr -n 4
```

This step can take some time depending on the number of cores you have allocated for the job. Once the job has finished, you can move on to the manual step to generate the training data.

### Training data annotation (Step 2)

You now need to annotate manually a series of peaks which will then be used to train the SVM classifiers and subsequently optimise the peak picking algoritm parameters to best fit your data. This task usually take several hours, but you only need to do that once per instrument/protocol. Once trained, the SVM classifiers can be reused on other datasets generated with the same instrument and data acquisition method.

To start the classification run the following command, you can add the flag `--help` to learn more:

```
../../run_WiPP.sh an
```

The script opens a simple visualization tool using the default pdf viewer, and will wait for to assign a class to the peak. Seven different classes are available for you to choose from.
Once you have annotated the required amount of peaks for each algorithm (1200 by default), the tool will automatically close. You are now ready to launch the last part of the pipeline.

> ### Note
> The default parameters are only taken if they are not defined in the `config.yaml` file. Chances are that you used example config file that we provided as a template, make sure to adapt the parameters properly. For instance, the number of peaks to annotated is set to 25 per sample in the example file, make sure to change this parameter before running this step.

### Optimisation and high confidence peak set generation (Step 3)

Many sequential substep are in fact happening during this final step of the pipeline such as classifier hyperparameter optimisation, peak detection algorithms parameter optimisation, peak detection on the full dataset, output classification and result integration.
Run it with the following command and adjust the number of cores using the inline parameter -n <CORES>:

```
../../run_WiPP.sh pp -n 4 
```

## Results

All result files are created in a new subfolder named 04_Final_results. 

Two types of results are available, first a csv file (wide format) contains the feature table, every row represent a peak and each column a specific information such as the retention time, retention index or the spectrum (peak mz:peak area) for a specific sample. Missing peaks are represented by missing values in the corresponding sample column.

The second output is a .msp file which contains the spectra of every peak reported in the .csv file and can be used for identification purposes (Using a dedicated GC-MS peak identification tool).

A .csv/.msp pair is available for every biological condition as well as all samples together.

## Run **WiPP** on an HPC

This section is dedicated to advanced users who wish to set up **WiPP** on a computing cluster for routine use. 

Running **WiPP** on a cluster requires the user to be logged into an HPC node with access to a SLURM scheduler. (i.e. jobs are submitted using 
`sbatch`)

To submit a job to the HPC scheduler include the `-x` flag, plus two additional optional parameters:
  * -x, --external : Submit the job to the scheduler using `sbatch` (REQUIRED)
  * -g <GIGS>, --gigs-per-cpu <GIGS> : Used to set the `sbatch` parameter `--mem-per-cpu` (OPTIONAL: default 6) 
  * -n <NODES>, --nodes <NODES> : Used to set the `sbatch` parameter `--cpus-per-task` (OPTIONAL: default 4)

e.g.
```bash
cd projects/my_project
../../run_WiPP.sh pp -x -g 10 -n 7
```

Once your job is running on the cluster, the system out logging messages are written to the file `slurm_log/wipp_<pipeline-step>_<project-name>_<cluster-job-number>.log`  
e.g. `projects/my_project/slurm_log/wipp_pp_installation_test-3089255.log`

*For information on how to monitor running jobs or troubleshoot scheduling errors, please refer to your local HPC user documentation*
