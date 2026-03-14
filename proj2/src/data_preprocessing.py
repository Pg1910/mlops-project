import os 
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')

# ensure the logs dir exists
log_dir = 'logs'
os.makedirs(log_dir , exist_ok= True)

# setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir ,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter( '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """ does transformation based by converting it to lowercase , tokenizing , remove stopwords and punctuations , and stemming"""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.tokenize.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_df (df , text_column = 'text', target_column = 'target'):
    """ preprocess the data frame  by encoding the target column  , removing duplicates ,  and transforming the text columns """
    try:
        logger.debug('starting preprocessing for Dataframe')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('target column encoded')
        # remove duplicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('duplicates removed')

        # apply text transformation to the sopecified  texxt columsn 
        df.loc[: , text_column] = df[text_column].apply(transform_text)
        logger.debug('text column transformed ')
        return df
    except KeyError as e:
        logger.error('column  not found : %s ',e)
        raise

def main(text_column = 'text', target_column = 'target'):
    """ main function to load raw data , preprocess it ,and save  the processed data"""
    try :
        # fetch the data from rae 
        train_data = pd.read_csv('proj2/data/raw/train.csv')
        test_data = pd.read_csv('proj2/data/raw/test.csv')
        logger.debug('Data loaded properly')

        #transform the data
        train_processed_data = preprocess_df(train_data, text_column , target_column)
        test_processed_data = preprocess_df(test_data, text_column , target_column)
        
        # store the data inside dataprocessed 
        data_path = os.path.join("proj2/data","interim")
        os.makedirs(data_path , exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path , "train_processed.csv"),index = False)
        test_processed_data.to_csv(os.path.join(data_path , "test_processed.csv"), index = False)
        logger.debug("preprcocessed data saved to %s",data_path)

    except FileNotFoundError as e:
        logger.error('file not found : %s', data_path)
    except pd.errors.EmptyDataError as e:
        logger.error('No data : %s' , e )
    except Exception as e:
        logger.error('failed to complete the data transformations  process :%s' , e)
        print(f"error :{e}")

if __name__ == '__main__':
    main()