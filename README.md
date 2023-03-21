# Online Processing of Vehicular Data on the Edge Through an Unsupervised TinyML Regression Technique


## Authors:  Pedro Andrade, Ivanovitch Silva, Marianne Diniz, Thommas Flores, Daniel G. Costa, and Eduardo Soares.


## Environment Setup
- First, start by closing the repository:
```bash

```
-   Start by installing  `WandB`  if you don't have it:

```bash
!pip3 install wandb -qU
```

To remove an environment in your shell/prompt, run the following command:

```bash
!pip3 install codecarbon
```

To list all available environments, run the following command:

```bash
conda env list
```

To list all packages installed in a environment, run the following command:

```bash
conda list
```

***

## Steps for Definition of Aforementioned Methodology

We will use DVC to manage and version the data processes that produce our results. This mechanism allows you to organize the project better and reproduce your workflow/pipeline and results more quickly.

The following steps are considered:
  1. ``data acquisition``;
  2. ``sweep variables definition``;
  3. ``WandB sweep``;
  4. ``data analysis``.

### 1) Data Acquisition

The updated raw data could be collected by the Jupyter Notebook or Python Script, that are contained into the folder ``data_collection``. Note that the updated raw datasets must be in the folder ``data_analysis_pipeline/data/raw``.

If you choose to include the updated data, it is necessary to track and version the input files. So, you need to run the following commands:

```bash
dvc add data_analysis_pipeline/data/raw/*.csv
git add data_analysis_pipeline/data/.gitignore
git add data_analysis_pipeline/data/raw/*.csv.dvc
```

If you want to use the version used, the raw data are available on [Mendeley Data](https://doi.org/10.17632/rwfd6p6xsd.1). Also, they are contained into the folder ``data_collection/raw_data``.



**Code 01**: TEDA algorithm [![Open in PDF](https://img.shields.io/badge/-PDF-EC1C24?style=flat-square&logo=adobeacrobatreader)](https://github.com/pedrohmeiraa/TEDA/blob/f498fade1b038fc22f8048d2a54fea2721b22483/1%20-%20TEDA%20Algorithm.pdf)
- TEDA, Introduction, Advantages x Disavantages, Applications, and Algorithm demonstration.



- [1] :books: Angelov, Plamen. (2014). *Anomaly detection based on eccentricity analysis*. 1-8. 10.1109/EALS.2014.7009497 [[Link]](https://www.researchgate.net/publication/301411485_Anomaly_detection_based_on_eccentricity_analysis). 

- [2] :books: **Andrade, P.**; Silva, I.; Silva, M.; Flores, T.; Cassiano, J.; Costa, D.G. *A TinyML Soft-Sensor Approach for Low-Cost Detection and Monitoring of Vehicular Emissions*. Sensors 2022, 22, 3838. doi: 10.3390/s22103838 [[Link]](https://www.mdpi.com/1424-8220/22/10/3838)

## Citations

[![DOI:10.3390/s22103838](https://zenodo.org/badge/DOI/10.3390/s22103838.svg)](https://www.mdpi.com/1424-8220/22/10/3838)

### How does it cite?

**Andrade, P.**; Silva, I.; Silva, M.; Flores, T.; Costa, D.G. Soares, E.; _Online Processing of Vehicular Data on the Edge Through an Unsupervised TinyML Regression Technique_. ACM TECS 2023, DOI: xx.xxxx/X (in submission process).

### How does the article download?

You can download the article from this [link](link).

### How to download the data?

You can download the data in this repository or clicking [here](https://github.com/pedrohmeiraa/TEDA/blob/f498fade1b038fc22f8048d2a54fea2721b22483/1%20-%20TEDA%20Algorithm.pdf).
