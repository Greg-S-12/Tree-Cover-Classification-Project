{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dummies_back_to_categorical(data,range_of_columns,categorical_column_name):\n",
    "    \n",
    "    # Create 2 lists of column names \n",
    "    columns_to_convert = data.iloc[:,range_of_columns].columns\n",
    "    list_of_columns = list(columns_to_convert)\n",
    "    \n",
    "    # Cycle through each dummy column name and create a list of separate dataframes\n",
    "    # for each dummy column where the dummy is true\n",
    "    iteration = -1    \n",
    "    for col in columns_to_convert:\n",
    "        iteration+=1\n",
    "        list_of_columns[iteration] = data.loc[data[col]==1,:]\n",
    "    list_of_dataframes = list_of_columns\n",
    "    \n",
    "    # Now cycle through our list of dataframes adding the new categorical column\n",
    "    # and assigning a number for each dummy variable from 1 to max in our range\n",
    "    iteration = 0\n",
    "    for i in list_of_dataframes:\n",
    "        iteration+=1\n",
    "        i[categorical_column_name] = iteration\n",
    "    \n",
    "    \n",
    "    data_no_dummies = pd.concat(list_of_dataframes).reset_index()\n",
    "    data_no_dummies = data_no_dummies.drop(columns=['index',columns_to_convert])\n",
    "    \n",
    "    return data_no_dummies\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:learn-env] *",
   "language": "python",
   "name": "conda-env-learn-env-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
