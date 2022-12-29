# Disaster Response Pipeline Project


## Overview
The scope of this project is to build a machine learning pipeline that classififes emergency disaster messages.
A classifier was trained using real world data to aim at classifying messages and the predictions are made available 
in a web application through an API developed in Flask.


## Installations
All the necessary libraries can be found in the file 'requirements.txt' or the whole conda environment can be created
by using the 'environment.yml' file.


## File Description

app

- templates
  - master.html                                        # front-end of the web page
  - go.html                                            # front-end of the web page
- run.py                                               # Flask framework application code

data

- disaster_categories.csv                              # real world data
- disaster_messages.csv                                # real world data
- process_data.py                                      # data cleaning pipeline
- InsertDatabaseName.db                                # database to save cleaned data 

models

- train_classifier.py                                  # machine learning pipeline
- classifier.pkl                                       # saved classification model


README.md

requirements.txt

environment.yml


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - To run the web app `python run.py`

