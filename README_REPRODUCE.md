# quantinar-rep

## Installation

- Create a virtual environment with conda and python3.9:
```bash
conda create -n ENV_NAME python=3.9
```

- Then, at the root (quantinar-rep), install the package with requirements in 
  setup.py with:
```bash
pip install .
```

Do not forget to upgrade the package after you implemented your changes with:
```bash
pip install . --upgrade
```

## Data

The data is available as a zip at: https://drive.google.com/file/d/1hzLSxiQaGwxQAO_xNFQvE6nBt0N3SNzb/view?usp=sharing 


## Scripts

In order to reproduce the results of the paper available at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4275797,
 just run the scripts in the `scripts` folder:

- `time_pagerank.py`: This script is used to compute the pagerank scores 
  for the given data. Please check the documentation inside the script.
- `analysis.py`: This script is used to produce the plots and statistics 
  presented in the paper. Please check the documentation in the script.