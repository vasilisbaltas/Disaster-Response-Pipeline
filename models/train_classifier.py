import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle



def load_data(database_filepath):
    """ Function that loads data from database and returns features and target
    
    """ 
    
    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('Emergency_Messages', engine)
    df = df.drop(columns = ['id','original'], axis=1)
    
    # drop rows with NaN messages
    df = df.dropna(subset=['message'])
    
    # our dataset still contains some null values to drop
    df = df.dropna()
    
    # we can observe that the 'related' category also contains double's(2) that does not make sense - we will turn this 2s     # to 1s
    df.loc[df.related==2, 'related'] = 1
    
    X = df['message']
    Y = df.drop(columns=['message','genre'], axis=1)
    
    return X, Y, Y.columns.tolist()
    

def tokenize(text):
    """ Function that tokenizes and lemmatizes text

    :param text:     input text to be processed(str)
    :return:         cleaned_tokens(str)
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Text normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenization
    words = word_tokenize (text)
    # lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word).strip() for word in words if word not in stop_words]

    return cleaned_tokens



def build_model():
    """ Function that builds RandomForest classifier with Grid Search
    
    :return:      Grid search Pipeline
    
    """
    
    pipeline = Pipeline([
     ('vect', CountVectorizer(tokenizer=tokenize)),
     ('tfidf', TfidfTransformer()),
     ('clf', RandomForestClassifier(random_state=33))
    ])
    
    parameters = {
    'clf__n_estimators' :[50,100,200],
    'clf__max_depth'        :[10,20,None]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """ Function that makes predictions and prints precision,recall,f1_score for each target category
    
    :param model:               trained classifier
    :param X_test:              test dataset features (df or numpy array)
    :param Y_test:              test dataset target categories (df or numpy array)
    :param category_names:      the names of target categories (list)
    :return:                    None
    
    """
        
    cv_preds = model.predict(X_test)
    pred_cols = category_names
    for i,col in enumerate(pred_cols):
        print(col.upper()+'\n', classification_report(Y_test[col].values, cv_preds[:,i]))

        

def save_model(model, model_filepath):
    """ Function that saves the classifier in a pickle format
    
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


    
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
