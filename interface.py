import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load data function
def load_data():
    df = pd.read_csv('Use_case_2_Dataset.csv')
    return df

# Data encoding function
def encode_data(df):
    cdf = df[['domain', 'hostname', 'serialnumber', 'isvirtual', 'tokenid', 'partitionid']]
    cdf["partitionid"] = cdf["partitionid"].str.strip().replace({
        'plkcsw01-vc:BUILTIN:BUILTIN': 1, 'pdqreq07-vc:BUILTIN:BUILTIN': 2, 
        'mlkewx05-vc:BUILTIN:BUILTIN': 3, 'thqlm09-vc:BUILTIN:BUILTIN': 4,
        'gbsabcd.mgmt.xyz.com:9013187907:9013119222': 5, 'gb1234.mgmt.xyz.com:9013187907:9013118765': 6
    })
    cdf['domain'] = cdf['domain'].str.strip().replace({
        'perf.bmc.com': 1, 'perf.area2.com': 2, 'perf.test.xyz.com': 3, 'drz.test.pc.com': 4
    })
    
    # Handle tokenid
    cdf['tokenid'].fillna(-1, inplace=True)
    token_values = cdf['tokenid'].unique()
    tokens = {val: 0 if val == -1 else 1 for val in token_values}
    cdf['tokenid'].replace(tokens, inplace=True)

    # Handle hostname
    cdf['hostname'].fillna(-1, inplace=True)
    hostnames = cdf['hostname'].unique()
    hnames = {val: 0 if val == -1 else 1 for val in hostnames}
    cdf['hostname'].replace(hnames, inplace=True)

    # Handle serialnumber
    cdf['serialnumber'].fillna(-1, inplace=True)
    serialnumbers = cdf['serialnumber'].unique()
    snumbers = {val: 0 if val == -1 else 1 for val in serialnumbers}
    cdf['serialnumber'].replace(snumbers, inplace=True)

    # Include the target column
    cdf['ReconciliationRuleId'] = df['ReconciliationRuleId']
    
    cdf.fillna(-1, inplace=True)
    return cdf

# Model training and prediction function
def train_model(cdf):
    X = cdf.drop(['ReconciliationRuleId'], axis=1)
    y = cdf['ReconciliationRuleId']
    unique_values = y.unique()
    rule_ids = {val: i for i, val in enumerate(unique_values)}
    y.replace(rule_ids, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_clf = XGBClassifier()
    xgb_clf.fit(X, y)
    
    return xgb_clf

# Streamlit app flow
try:
    df = load_data()
    if df is not None:
        cdf = encode_data(df)
        xgb_clf = train_model(cdf)
        
        st.write("Please enter the following details. Note:")
        st.markdown("""
        - **Domain**: Enter 1 for `perf.bmc.com`, 2 for `perf.area2.com`, 3 for `perf.test.xyz.com`, 4 for `drz.test.pc.com`
        - **Partition ID**: Enter 1 for `plkcsw01-vc:BUILTIN:BUILTIN`, 2 for `pdqreq07-vc:BUILTIN:BUILTIN`, 3 for `mlkewx05-vc:BUILTIN:BUILTIN`, 4 for `thqlm09-vc:BUILTIN:BUILTIN`, 5 for `gbsabcd.mgmt.xyz.com:9013187907:9013119222`, 6 for `gb1234.mgmt.xyz.com:9013187907:9013118765`.
        - Other fields: Enter 1 for known values, 0 for unknown.
        """)


        # Wrap the input in a form
        with st.form("input_form"):
            #st.write("Enter the following details:")
            domain_input = st.text_input("Domain (1-4)", "")
            hostname_input = st.text_input("Hostname (1 for known, 0 for unknown)", "")
            serialnumber_input = st.text_input("Serial Number (1 for known, 0 for unknown)", "")
            isvirtual_input = st.text_input("Is Virtual (1 for yes, 0 for no)", "")
            tokenid_input = st.text_input("Token ID (1 for known, 0 for unknown)", "")
            partitionid_input = st.text_input("Partition ID (1-6)", "")
            
            # Submit button within the form
            submit_button = st.form_submit_button("Predict Priority")
        
        if submit_button:
            try:
                # Ensure all inputs are filled before proceeding
                if all([domain_input, hostname_input, serialnumber_input, isvirtual_input, tokenid_input, partitionid_input]):
                    # Create a DataFrame for user input
                    user_input = pd.DataFrame({
                        'domain': [int(domain_input)],
                        'hostname': [int(hostname_input)],
                        'serialnumber': [int(serialnumber_input)],
                        'isvirtual': [int(isvirtual_input)],
                        'tokenid': [int(tokenid_input)],
                        'partitionid': [int(partitionid_input)]
                    })

                    # Predict the reconciliation rule priority
                    probabilities = xgb_clf.predict_proba(user_input)
                    predicted_class = xgb_clf.predict(user_input)

                    # Rename rules and sort by probabilities (same as before)
                    rule_names = {0: 100, 1: 110, 2: 120, 3: 130}
                    sorted_rules = sorted(enumerate(probabilities[0]), key=lambda x: x[1], reverse=True)

                    # Display the results
                    st.write(f"Predicted Reconciliation Rule ID: {rule_names[predicted_class[0]]}")
                    st.write("Priority order of reconciliation rules based on probabilities (sorted):")
                    for rule, prob in sorted_rules:
                        st.write(f"Rule {rule_names[rule]}: {prob:.5f}")
                else:
                    st.warning("Please fill out all fields before predicting.")

            except ValueError:
                st.error("Please enter valid numeric values.")
except Exception as e:
    st.error(f"Error: {e}")