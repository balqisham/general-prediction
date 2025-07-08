#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
    st.title('💲Cost Prediction RT2025💲')
    
    # File upload
    st.sidebar.header('Upload Data')
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read and display the data
        df = pd.read_csv(uploaded_file)
        
        # Data Overview Section
        st.header('Data Overview')
        st.write('Dataset Shape:', df.shape)
        st.dataframe(df.head())

        # Target selection
        st.subheader('Select Target Perimeter to Predict')
        target_column = st.selectbox('Select the target column (what you want to predict)', df.columns)
        
        # Prepare the data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Model Training Section
        st.header('Model Training')
        
        test_size = st.slider('Select test size (0.0-1.0)', 0.1, 0.5, 0.2)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Model Performance Section
        st.header('Model Performance')
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('RMSE', f'{rmse:.2f}')
        with col2:
            st.metric('R² Score', f'{r2:.2f}')
        
        # Visualization Section
        st.header('Data Visualization')
        
        # Correlation Matrix
        st.subheader('Correlation Matrix')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig)
        plt.close()
        
        # Feature Importance
        st.subheader('Feature Importance')
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance')
        st.pyplot(fig)
        plt.close()
        
        # Scatter Plots
        st.subheader('Feature vs Target Scatter Plots')
        feature_to_plot = st.selectbox('Select feature for scatter plot', X.columns)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=feature_to_plot, y=target_column)
        plt.xlabel(feature_to_plot)
        plt.ylabel(f'{target_column} (Target)')
        st.pyplot(fig)
        plt.close()
        
        # Prediction Section
        st.header('Make New Predictions')
        st.write('Enter values for prediction:')
        
        # Create input fields for each feature
        new_data = {}
        for column in X.columns:
            new_data[column] = st.number_input(f'Enter {column}', 
                                               value=float(X[column].mean()),
                                               step=float(X[column].std()) if X[column].std() > 0 else 1.0)
        
        if st.button('Predict'):
            # Prepare the input data
            input_data = pd.DataFrame([new_data])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = rf_model.predict(input_scaled)
            
            st.success(f'Predicted {target_column}: {prediction[0]:,.2f}')

if __name__ == '__main__':
    main()


# In[ ]:




