
import pandas as pd
from ..logging.logging import logging_decorator

@logging_decorator
def load_and_preprocess_data(data_path):
    
    # Import the data from 'real_estate.csv'
    df = pd.read_csv(data_path)

    # Impute missing values in basement
    df.fillna({'basement':0}, inplace=True)
    df['basement']=df['basement'].astype(int)        

    # Remove observations with lot_size greater than 500,000 sqft.
    df=df[df.lot_size<500000]
    return df
