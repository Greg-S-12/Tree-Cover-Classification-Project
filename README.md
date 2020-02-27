# Tree-Cover-Classification-Project

This project aims to solve a multi-class classification problem (7 classes) from an imbalanced dataset of 566K rows. 
Dataset: https://www.kaggle.com/c/forest-cover-type-prediction/overview

In this project we first conduct some EDA and create some visualizations to get a clearer picture of the data and the features that exist within it.
We then try multiple machine learning models (Logistic Regression, SVM, Decision Trees, Randomm Forests) to achieve the best weighted F1 score.
We used an one-vs-rest approach to classify each tree type and correctly classify 99% of the entries in our test dataset.
