# Omics Toolbox

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project is a streamlined pipeline to use the multiomics tools developed by the Krishnaswamy Lab.

## Setting Up the Environment

To set up the `omics_toolbox` environment, follow these steps:

1. Run the following command to create the environment:
    ```bash
    conda env create --file environment.yml
    ```

    This will install all the required packages and create a Conda environment named `omics_toolbox`.

2. Activate the environment:
    ```bash
    conda activate omics_toolbox
    ```

3. Ensure that the `omics_toolbox` environment is selected as the kernel for all Jupyter notebooks. You can do this by running:

You're now ready to use the `omics_toolbox` environment for your analyses.

> **Note:** Some packages in this project utilize PyTorch and require CUDA for model computations. If you encounter issues, they are often due to incompatibilities between the CUDA version and the package dependencies. Ensure that your CUDA version is compatible with the installed PyTorch and related packages. Refer to the [PyTorch CUDA Compatibility Guide](https://pytorch.org/get-started/previous-versions/) for more details.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         omics_toolbox and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── omics_toolbox   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes omics_toolbox a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

