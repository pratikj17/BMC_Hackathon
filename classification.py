import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
from xgboost import XGBClassifier
import seaborn as sns

def load_data():
    uploaded_file = st.file_uploader("Choose a file", type={"csv", "txt"})
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

# Data encoding function
def encode_data(df):
    cdf = df[['domain', 'hostname', 'serialnumber', 'isvirtual', 'tokenid', 'partitionid']]

    # Replaced partitionid with integers
    cdf["partitionid"] = cdf["partitionid"].str.strip().replace({
        'plkcsw01-vc:BUILTIN:BUILTIN': 1, 'pdqreq07-vc:BUILTIN:BUILTIN': 2, 
        'mlkewx05-vc:BUILTIN:BUILTIN': 3, 'thqlm09-vc:BUILTIN:BUILTIN': 4,
        'gbsabcd.mgmt.xyz.com:9013187907:9013119222': 5, 'gb1234.mgmt.xyz.com:9013187907:9013118765': 6
    }).fillna(-1).astype(int)

    # Replaced domain with integers
    cdf['domain'] = cdf['domain'].str.strip().replace({
        'perf.bmc.com': 1, 'perf.area2.com': 2, 'perf.test.xyz.com': 3, 'drz.test.pc.com': 4
    }).fillna(-1).astype(int)

    # Handled tokenid
    cdf['tokenid'].fillna(-1, inplace=True)
    token_values = cdf['tokenid'].unique()
    tokens = {val: 0 if val == -1 else 1 for val in token_values}
    cdf['tokenid'].replace(tokens, inplace=True)

    # Handled hostname
    cdf['hostname'].fillna(-1, inplace=True)
    hostnames = cdf['hostname'].unique()
    hnames = {val: 0 if val == -1 else 1 for val in hostnames}
    cdf['hostname'].replace(hnames, inplace=True)

    # Handled serialnumber
    cdf['serialnumber'].fillna(-1, inplace=True)
    serialnumbers = cdf['serialnumber'].unique()
    snumbers = {val: 0 if val == -1 else 1 for val in serialnumbers}
    cdf['serialnumber'].replace(snumbers, inplace=True)

    # Handled null values in isvirtual
    cdf['isvirtual'] = cdf['isvirtual'].fillna(-1).astype(int)

    # Included the target column
    cdf['ReconciliationRuleId'] = df['ReconciliationRuleId']
    
    return cdf

# Model training and prediction function
def train_model(cdf):
    
    # Train-Test-Split
    X = cdf.drop(['ReconciliationRuleId'], axis=1)
    y = cdf['ReconciliationRuleId']
    unique_values = y.unique()
    rule_ids = {val: i for i, val in enumerate(unique_values)}
    y.replace(rule_ids, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Fitting
    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb_clf.predict(X_test)
    probabilities = xgb_clf.predict_proba(X_test)
    
    # Adding probabilities to the DataFrame
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
    for i, rule_id in enumerate(unique_values):
        y_test_df[f'Probability for Rule ID {rule_id}'] = probabilities[:, i]
    y_test_df['Predicted Rule ID'] = y_pred
    
    return y_test_df, y_pred

# Function to calculate priority based on the sum of predictions
def calculate_priority(y_test_df, y_pred):
    
    # Created a DataFrame to hold the prediction results
    priority_df = pd.DataFrame({
        'ReconciliationRuleId': y_test_df.index,
        'predicted': y_pred
    })
    
    priority_sum = priority_df.groupby('predicted').size().reset_index(name='total_predictions')
    
    priority_sum_sorted = priority_sum.sort_values(by='total_predictions', ascending=False)
    
    priority_mapping = {0: 100, 1: 110, 2: 120, 3: 130, 4:140}
    priority_sum_sorted['priority'] = priority_sum_sorted['predicted'].replace(priority_mapping)
    
    return priority_sum_sorted[['priority', 'total_predictions']]

try:
    df = load_data()
    if df is not None:
        cdf = encode_data(df)
        y_test_df, y_pred = train_model(cdf)
        
        st.write("Predictions and Probabilities:")
        st.write(y_test_df)
        
        priority_df = calculate_priority(y_test_df, y_pred)
        st.write("Priority based on total predictions for each rule:")
        st.write(priority_df)
except Exception as e:
    st.error(f"Error: {e}")
