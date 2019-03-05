import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from sqlalchemy import create_engine
import pandas as pd
import numpy as np

def load_data(database_filepath):
    """
    Load the data from a database
    
    args:
    database_filepath: .db file from which to load the data
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_query('SELECT * FROM disaster', engine)
    X = df["message"].values
    temp_df = df.drop(['message', 'original', 'genre', 'id'], axis=1)
    Y = temp_df.values
    return X, Y, temp_df.index.values

def tokenize(text):
    """
    Tokenize the input text
    
    Args:
        text: list. 
            Input to tokenize
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a model to predict the class of a message.
    The model is a nlp pipeline made of tfidf and randomclassifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=100, min_samples_split=10)))
    ])
    parameters = parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [1, 5, 10, 15],
        'clf__estimator__min_samples_split': [10, 20, 30, 40],
        'clf__estimator__max_depth': [50, 100, 200]
    }

    return GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print the accuracy and recall of the model as regards all targets.

    Args:
        model: model to evaluate
        X_test: test features
        Y_test: test labels
"""
    Y_pred = model.predict(X_test)
    columns = Y_pred.shape[1]
    for c in range(columns):
        target_names = np.unique(Y_test[:, c])
        print(target_names)
        print(classification_report(Y_pred[:, c], Y_test[:, c]))


def save_model(model, model_filepath):
    """
    Save a model to a filepath.

    Args:
        model: model to save.
        model_filepath: .flk filepath to save the model.
    """
    joblib.dump(model, model_filepath) 

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()