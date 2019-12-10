"""
This module is for your data cleaning.
It should be repeatable.

## PRECLEANING
There should be a separate script recording how you transformed the json api calls into a dataframe and csv.

## SUPPORT FUNCTIONS
There can be an unlimited amount of support functions.
Each support function should have an informative name and return the partially cleaned bit of the dataset.
"""
import pandas as pd


def rename_columns(dataframe, current_columns, new_columns):
       
    df=dataframe
    
    for column, name in list(zip(current_columns,new_columns)):
        df.rename(columns={column:name},inplace=True)
    return df

def drop_columns(dataframe,columns_to_drop):
    df=dataframe
    return df.drop(columns=columns_to_drop, inplace=True)


def support_function_three(example):
    pass






def full_clean(dataframe, current_columns, new_columns, columns_to_drop):
    
    df_clean1 = rename_columns(dataframe, current_columns, new_columns)
    df_cleaned = drop_columns(df_clean1, columns_to_drop)
    return df_cleaned
    
    
    """
    This is the one function called that will run all the support functions.
    Assumption: Your data will be saved in a data folder and named "dirty_data.csv"

    :return: cleaned dataset to be passed to hypothesis testing and visualization modules.
    """
    dirty_data = pd.read_csv("./data/dirty_data.csv")

    cleaning_data1 = support_function_one(dirty_data)
    cleaning_data2 = support_function_two(cleaning_data1)
    cleaned_data= support_function_three(cleaning_data2)
    cleaned_data.to_csv('./data/cleaned_for_testing.csv')
    
    return cleaned_data