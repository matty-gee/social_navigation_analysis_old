# Social Navigation Task usage notes

## Installation

### (1) Clone repository locally

Download and unzip as a local directory

### (2a) Set up a conda environment (recommended)

In terminal, navigate to the unzipped directory and run:

```bash
conda env create -f env.yml # will create a conda environment called 'social_navigation_analysis'.... may take a minute 
conda activate social_navigation_analysis # activates the environment, so have access to packages etc
```

### (2b) pip install (not recommended)
```bash
pip install --user --upgrade git+https://github.com/matty-gee/social_navigation_analysis.git
```

## Usage

If cloned, add the directory with the cloned repository into the system path, e.g.: 

```python
# add directory into python system path
import sys
sys.path.insert(0, /path/to/social_navigation_analysis/social_navigation_analysis')
```

Then, if pip or cloned, import the module 

```python
import social_navigation_analysis as snt
```

## Functions

| Function | Descriotion |
| :----: | --- |
| `function` | some function |


## Contributing


## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author

Matthew Schafer

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
