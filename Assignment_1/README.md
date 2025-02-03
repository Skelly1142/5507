Preprocessing script 
logistic regression model 
Packages/Dependencies 
Usage 
License 

1. Preprocessing script 

    a) Impute Missing Values 
        
        A function to fill in missing data from the dataset using one of three strategies: Mean, Median, or Mode 

        def impute_missing_values(clean_data, strategy='mean'):

        requirements = pandas dataframe

    b)  Removal of Duplicate Rows

        A function to remove any duplicate rows from the dataset 

        def remove_duplicates(clean_data):


    c) Normalize Numerical Data 

        A function to normailze numerical features within the dataset using Min-Max scaling 

        def normalize_data(clean_data, method='minmax'):
    
    d) Removal of redundant features 

        A function to remove redundant features based on correlation. Features with correlation above a threshold will be removed from the data. Threshold used was 0.9 

        def remove_redundant_features(clean_data, threshold=0.9):

2. Logistic Regression Model

    a) The simple_model function will implement a logistic regression model on the preprocessed data. 

        def simple_model(input_data, split_data=True, scale_data=False, print_report=False):

3. packages/Dependencies 

    a) scikit-learn 
    b) pandas
    c) numpy 


4. Usage of Functions in a Notebook 

    # Import necessary modules
    import data_preprocessor as dp
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # 1. Load the dataset
    messy_data = pd.read_csv('../Data/messy_data.csv')
    clean_data = messy_data.copy()



    # 2. Preprocess the data
    clean_data = dp.impute_missing_values(clean_data, strategy='mean')
    clean_data = dp.remove_duplicates(clean_data)
    clean_data = dp.normalize_data(clean_data)
    clean_data = dp.remove_redundant_features(clean_data)

    # 3. Save the cleaned dataset
    clean_data.to_csv('../Data/clean_data.csv', index=False)


    # steps to ensure data and columns are ready for modelling 
    target_column = clean_data.columns[0]
    if clean_data[target_column].dtype != 'object' and clean_data[target_column].dtype != 'category':
    # If target is continuous, convert to binary or categorical labels (example)
    threshold = clean_data[target_column].median()
    clean_data[target_column] = (clean_data[target_column] > threshold).astype(int)  # Binary labels

    # 4. Train and evaluate the model
    dp.simple_model(clean_data)


5. Licensing 

This project is licensed under the MIT License 






    



