# README #

##What is this repository for?

CIL Repository for Collaborative Filtering:  
* Kaggle: https://inclass.kaggle.com/c/cil-collab-filtering-2017  
* Repo to exercise 3:  https://github.com/dalab/lecture_cil_public/tree/master/exercises/ex3

##How do I get set up?
###How to run:
####Local:
1. Clone to local disk

#####with IDE:
2. Install PyCharm (https://www.jetbrains.com/pycharm/)
3. Open the folder `cil2017` as project and run `Start.py` with the config file as parameter.

#####with script:
2. Navigate to `collab_filtering/scripts`
3. Run `start_local.sh`

####On Euler:
1. Adjust the NETHZ variable in `start_remote.sh`
2. Run it   


###Dependencies
The easiest way is to install anaconda for Python 3: https://www.continuum.io/downloads

##Structure

.  
├── `collab_filtering`  
│   ├── `config`: config file  
│   ├── `in`: input data  
│   ├── `pdfs`: pdfs related to the project  
│   └── `scripts`: python and shells scripts  
├── `out`: predictions and log  
└── `report`: report-related files


##How do I generate the data used in the report? 
The current `config.json` is the one to generate the kaggle submission.  
To run local tests rename the config file associated with the desired test to `config.json`.  
1. `config_test_*.json` files are used to generate the results in table 2.  
2. `config_fig_k.json` file is used to generate results for figure 1.  
3. `config_fig_lambda_d.json` file is used to generate results for figure 2.  
4. `config_submission.json` file is used to generate submission file for kaggle.