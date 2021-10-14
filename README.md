# README #

## What is this repository for?

Computational Intelligence Lab repository for the Collaborative Filtering project based on a 
[Kaggle competition](https://inclass.kaggle.com/c/cil-collab-filtering-2017) (Team 'Die 3 Weisen', 1st place).

## How do I get set up?

### Dependencies
Use pip to install:
- numpy
- sklearn
- statistics

### How to run:
#### Local:
1. Clone to local disk

##### with IDE:
2. Install [PyCharm](https://www.jetbrains.com/pycharm/)
3. Open the folder `cil2017` as project and run `Start.py` with the config file as parameter.

##### with script:
2. Navigate to `collab_filtering/scripts`
3. Run `start_local.sh`

#### On Euler:
1. Adjust the NETHZ variable in `start_remote.sh`
2. Run it   

## How do I generate the data used in the report? 
The current `config.json` is the one to generate the kaggle submission.  
To run local tests rename the config file associated with the desired test to `config.json`.  
1. `config_test_*.json` files are used to generate the results in table 2.  
2. `config_fig_k.json` file is used to generate results for figure 1.  
3. `config_fig_lambda_d.json` file is used to generate results for figure 2.  
4. `config_submission.json` file is used to generate submission file for kaggle.