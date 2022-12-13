# Social Navigation Task usage notes

## Installation

### (1) Clone repository locally

Download and unzip locally

### (2a: recommended) Set up a conda environment

From the directory you cloned: 

```bash
conda env create -f env.yml # will create a conda environment called 'social_navigation_analysis'.... may take a minute 
conda activate social_navigation_analysis # activates the environment, so have access to packages etc
```

### (2b: not recommended) pip install
```bash
pip install --user --upgrade git+https://github.com/matty-gee/social_navigation_analysis.git
```

## Usage

If cloned, add the directory with the cloned repository into the system path so your python can find the module, e.g.: 

```python
# add directory into python system path
import sys
sys.path.insert(0, /path/to/social_navigation_analysis/social_navigation_analysis')
# then import 
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
