# Code accompanying paper: Direct Amortized Likelihood Ratio Estimation

To run the code in this repository, first run:

```
pip install .
```

This repository contains:

* `./notebooks/quadcopter_example.ipynb`: This is an example notebook that runs all three likelihood ratio estimators (DNRE, BNRE, NRE) to both train and sample quadcopter designs.
* `./src/dnre/benchmark.py`: This code runs the benchmark using the following command:
    ```
    python src/dnre/benchmark.py --model_type dnre --task two_moons --save_dir ./benchmark_results/two_moons/dnre --device 0
    ```
    Where we have selected DNRE as the approach to perform grid search over in the above command. The code includes comments for all additional options.
    
* `./src/dnre/benchmark_evaluate.py`: This code evaluates the above best result from the grid search using the following command:
    ```
    python src/dnre/benchmark_evaluate.py --model_type dnre --task two_moons --path_dir ./benchmark_results/two_moons/dnre --device 0 --metric coverage
    ```
    Where we point to the same directory as above. This will evaluate the expected coverage, but there are also options for C2ST and the log posterior.
* `./data/data_dict_4490`: Contains the data from the quadcopter experiment.

## Acknowledgements

This project was supported by DARPA under the Symbiotic
Design for Cyber-Physical Systems (SDCPS) with contract
FA8750-20-C-0002.
The views, opinions and/or findings expressed
are those of the author and should not be interpreted as
representing the official views or policies of the Department
of Defense or the U.S. Government.