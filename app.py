import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
#from sklearn.preproces import  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_excel('media_ratings.xlsx')

# Replace NaN values with a placeholder string in relevant columns
data['Director'].fillna('Unknown', inplace=True)
data['Writer'].fillna('Unknown', inplace=True)
data['Producer'].fillna('Unknown', inplace=True)
data['Actors'].fillna('Unknown', inplace=True)  # Important for splitting

# Replace directors, writers, and producers with only 1 or 2 dramas with "Others"
min_contributions = 3  # Minimum number of contributions to not be considered "Others"

director_counts = data['Director'].value_counts()
writer_counts = data['Writer'].value_counts()
producer_counts = data['Producer'].value_counts()

data['Director'] = data['Director'].apply(lambda x: x if director_counts[x] >= min_contributions else 'Others')
data['Writer'] = data['Writer'].apply(lambda x: x if writer_counts[x] >= min_contributions else 'Others')
data['Producer'] = data['Producer'].apply(lambda x: x if producer_counts[x] >= min_contributions else 'Others')

# Remove producers with content getting position 2 times or less
min_producer_contributions = 3  # Minimum number of contributions to not be removed
data = data[data['Producer'].map(producer_counts) > min_producer_contributions]

# Split actors by comma and apply one-hot encoding
data['Actors'] = data['Actors'].str.split(', ')

# Create a set of all unique actors
all_actors = set(actor for actors_list in data['Actors'] for actor in actors_list)

# Create a new DataFrame with one column per actor, filled with 0 or 1
for actor in all_actors:
    data[actor] = data['Actors'].apply(lambda x: 1 if actor in x else 0)

# Select features and target variable
X = data[['Channel', 'Producer', 'Writer', 'Director'] + list(all_actors)]
y = data['Rating']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline (one-hot encode categorical features)
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Channel', 'Producer', 'Writer', 'Director'])
    ],
    remainder='passthrough'  # Actors are already encoded
)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Model selection dropdown
selected_model = st.selectbox('Select Regression Model', list(models.keys()))

# Define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', models[selected_model])
])

# Train the model
model.fit(X_train, y_train)

# Calculate R^2 score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title('Viewership Rating Prediction')

# Input for features
channel = st.selectbox('Channel', data['Channel'].unique())
producer = st.selectbox('Producer', data['Producer'].unique())
writer = st.selectbox('Writer', data['Writer'].unique())
director = st.selectbox('Director', data['Director'].unique())
actors_selected = st.multiselect('Actors', list(all_actors))

# Create input data for prediction
input_data = pd.DataFrame([[channel, producer, writer, director] + 
                           [1 if actor in actors_selected else 0 for actor in all_actors]],
                          columns=['Channel', 'Producer', 'Writer', 'Director'] + list(all_actors))

# Make prediction
rating_prediction = model.predict(input_data)

# Display y_pred and y_test in DataFrames
st.subheader('Predicted vs Actual Ad Rating:')
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(df_results)

# Display prediction and R^2 score
st.write('Predicted Rating:', rating_prediction[0])
st.write('R^2 Score:', r2)
sing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_excel('G:\ECOMMERCE ACCOUNTRS\Account\Monthly reporting\Month Wise Top Drmas data.xlsx', sheet_name='tARGET DATA')

# Replace NaN values with a placeholder string in relevant columns
data['Director'].fillna('Unknown', inplace=True)
data['Writer'].fillna('Unknown', inplace=True)
data['Producer'].fillna('Unknown', inplace=True)
data['Actors'].fillna('Unknown', inplace=True)  # Important for splitting

# Replace directors, writers, and producers with only 1 or 2 dramas with "Others"
min_contributions = 3  # Minimum number of contributions to not be considered "Others"

director_counts = data['Director'].value_counts()
writer_counts = data['Writer'].value_counts()
producer_counts = data['Producer'].value_counts()

data['Director'] = data['Director'].apply(lambda x: x if director_counts[x] >= min_contributions else 'Others')
data['Writer'] = data['Writer'].apply(lambda x: x if writer_counts[x] >= min_contributions else 'Others')
data['Producer'] = data['Producer'].apply(lambda x: x if producer_counts[x] >= min_contributions else 'Others')

# Remove producers with content getting position 2 times or less
min_producer_contributions = 3  # Minimum number of contributions to not be removed
data = data[data['Producer'].map(producer_counts) > min_producer_contributions]

# Split actors by comma and apply one-hot encoding
data['Actors'] = data['Actors'].str.split(', ')

# Create a set of all unique actors
all_actors = set(actor for actors_list in data['Actors'] for actor in actors_list)

# Create a new DataFrame with one column per actor, filled with 0 or 1
for actor in all_actors:
    data[actor] = data['Actors'].apply(lambda x: 1 if actor in x else 0)

# Select features and target variable
X = data[['Channel', 'Producer', 'Writer', 'Director'] + list(all_actors)]
y = data['Rating']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline (one-hot encode categorical features)
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Channel', 'Producer', 'Writer', 'Director'])
    ],
    remainder='passthrough'  # Actors are already encoded
)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Model selection dropdown
selected_model = st.selectbox('Select Regression Model', list(models.keys()))

# Define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', models[selected_model])
])

# Train the model
model.fit(X_train, y_train)

# Calculate R^2 score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title('Viewership Rating Prediction')

# Input for features
channel = st.selectbox('Channel', data['Channel'].unique())
producer = st.selectbox('Producer', data['Producer'].unique())
writer = st.selectbox('Writer', data['Writer'].unique())
director = st.selectbox('Director', data['Director'].unique())
actors_selected = st.multiselect('Actors', list(all_actors))

# Create input data for prediction
input_data = pd.DataFrame([[channel, producer, writer, director] + 
                           [1 if actor in actors_selected else 0 for actor in all_actors]],
                          columns=['Channel', 'Producer', 'Writer', 'Director'] + list(all_actors))

# Make prediction
rating_prediction = model.predict(input_data)

# Display y_pred and y_test in DataFrames
st.subheader('Predicted vs Actual Ad Rating:')
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(df_results)

# Display prediction and R^2 score
st.write('Predicted Rating:', rating_prediction[0])
st.write('R^2 Score:', r2)
