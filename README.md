## Smoky Mountain Data Challenge 2022 - Analysis of the Summit Login Nodes Usage Data

The associated code from the Smoky Mountains Conference Data Challenge 2022 used to parse, process, and clean the raw data from data challenge #2 - Analysis of the Summit Login Nodes Usage Data. https://smc-datachallenge.ornl.gov/summit-login-nodes/

The original dataset can be found here. https://doi.ccs.ornl.gov/ui/doi/386

Data was processed using a combination of Jupyter Notebook python scrips and standard python scripts. 

### Step 1: Processing the log data into 8 separate linux command datasets
- - -
Run the `process_log` notebook in the `notebooks` folder. This should take a while. Adjust the number of CPUs used for processing depending on the system.

### Step 2: Pruning the data
- - -
Run each notebook in `notebooks` corresponding to the dataset to be pruned.
