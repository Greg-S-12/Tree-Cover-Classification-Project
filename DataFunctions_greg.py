import pandas as pd


def dummies_back_to_categorical(data,range_of_columns,categorical_column_name):
    
    # Create 2 lists of column names 
    columns_to_convert = data.iloc[:,range_of_columns].columns
    list_of_columns = list(columns_to_convert)
    
    # Cycle through each dummy column name and create a list of separate dataframes
    # for each dummy column where the dummy is true
    iteration = -1    
    for col in columns_to_convert:
        iteration+=1
        list_of_columns[iteration] = data.loc[data[col]==1,:]
    
    list_of_dataframes = list_of_columns
    
#     # Now cycle through our list of dataframes adding the new categorical column
#     # and assigning a number for each dummy variable from 1 to max in our range
    iteration = 0
    for i in list_of_dataframes:
        iteration+=1
        i[categorical_column_name] = iteration
        
    data_concat = pd.concat(list_of_dataframes).reset_index()
    data_no_dummies = data_concat.drop(columns=columns_to_convert)
    data_no_dummies.drop(data_no_dummies.columns[0], axis = 1, inplace=True)
    
    return data_no_dummies
