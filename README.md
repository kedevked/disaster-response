# Disaster Response Pipeline Project

This project is part of Udacity Data-Science Nanodegree. It uses ETL pipeline to process the data and a machine learning pipeline to train and predict.

The machine learning pipeline is made of nlp using tf-idf and a randomclassifier. Multiple targets are predicted by using [multi output classifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)

### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
