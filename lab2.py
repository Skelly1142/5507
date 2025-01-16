import functionals as f
import pandas as pd
import numpy as np

# Load the dataset
path_to_file = 'messy_data.csv'
data = pd.read_csv(path_to_file)

# Display dataset information

data.head()
data.info()
data.describe()

# Run the simple model
f.simple_model(data)