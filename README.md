# AttentionGraphNodeIdenfitication

This is the experiment code for the paper "Identifying Informative Nodes in Attributed Spatial Sensor Networks using Attention for Symbolic Abstraction in a GNN-based Modeling Approach". It models an extrinsic regression task using different GCN-based models, one including a transformer. Further including LASA (an  attention-based abstraction technique) to find more relevant nodes.


### Data

The data (too big to host on github itself) can be downloaded at: https://zenodo.org/record/5767221 <br>
In the data folder, the input_ci.npy file should be placed <br>
The input_cw.npy file should be placed in data/othernetwork  <br>

### Dependencies installation guide

A list of all needed dependencies (other versions can work but are not guaranteed to do so):

Requirements for the installation guide (other version might or might not work as well):

- python==3.9.13
- conda==22.9.0
- pip==22.3

We suggest using a fresh conda environment and also to use mamba as on top solution to conda, else an dependency solve freeze can occur.<br>
Create a new environment and run the following lines:

mamba install -c conda-forge tensorflow-gpu==2.6.2

Install the following dependencies using pip install, one at a time (else dependency errors can occur):
- seml==0.3.7 
- spektral==1.2.0 
- cython==0.29.33
- uea_ucr_datasets==0.1.2
- pandas==1.4.4
- dill==0.3.6
- scikit-learn==1.1.2
- scipy==1.9.1
- pyts==0.12.0
- numpy==1.19.5
- gast==0.4.0 


### How to run

We have two options to run the experiment. Either just test out single configurations with an anaconda notebook or test out multiple parameter combinations over SEML experiments.

#### Just for testing

1. Go to evalExample.ipynb
2. Change parameters
3. Have fun

#### Multiple experiment settings with SEML

1. Set up seml with seml configure <font size="6">(yes you need a mongoDB server for this and yes the results will be saved a in separate file, however seml does a really well job in managing the parameter combinations in combination with slurm) </font>
2. Configure the yaml file you want to run. Probably you only need to change the number of maximal parallel experiments ('experiments_per_job' and 'max_simultaneous_jobs') and the memory and cpu use ('mem' and 'cpus-per-task').
3. Add and start the seml experiment. For example like this:
	1. seml eqModel add eqModel.yaml
	2. seml eqModel start
4. Check with "seml eqModel status" till all your experiments are finished 
5. Please find the results in the results folder. It includes a dict which can be explored with the code in eval.ipynb

## Cite and publications

This code represents the used model for the following publication:<br>
"Identifying Informative Nodes in Attributed Spatial Sensor Networks using Attention for Symbolic Abstraction in a GNN-based Modeling Approach" (TODO Link)

If you use, build upon this work or if it helped in any other way, please cite the linked publication.