import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import linregress
from sklearn.impute import KNNImputer
import io

# Set page config
st.set_page_config(
    page_title="Cost Prediction RT2025",
    page_icon="💲",
    initial_sidebar_state="expanded"
)

# Add this import for formatting axis ticks
from matplotlib.ticker import FuncFormatter

# Formatter function for human-readable axis labels
def human_format(num, pos=None):
    if num >= 1e9:
        return f'{num/1e9:.1f}B'
    elif num >= 1e6:
        return f'{num/1e6:.1f}M'
    elif num >= 1e3:
        return f'{num/1e3:.1f}K'
    else:
        return f'{num:.0f}'

# Function to format numbers with commas
def format_with_commas(num):
    return f"{num:,.2f}"

# Check if the user is authenticated with just a password
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if "password_correct" in st.session_state:
        return st.session_state["password_correct"]

    # Show input for password only.
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=password_entered)

    # Return False if the password is wrong or not yet validated.
    return False

def get_currency_symbol(df):
    """Extract currency symbol from column names or return default"""
    for col in df.columns:
        if 'RM' in col.upper():
            return 'RM'
        elif 'USD' in col.upper() or '$' in col:
            return 'USD'
        elif 'EUR' in col.upper() or '€' in col:
            return 'EUR'
        elif 'GBP' in col.upper() or '£' in col:
            return 'GBP'
    return 'RM'  # Default to RM

def format_currency(amount, currency='RM'):
    """Format amount as currency in millions"""
    return f"{currency} {amount:.2f} Mil"

def download_all_predictions():
    """Create a combined Excel file with all predictions separated by dataset in different sheets"""
    if not st.session_state['predictions'] or all(len(preds) == 0 for preds in st.session_state['predictions'].values()):
        st.sidebar.error("No predictions available to download")
        return
    
    # Create a buffer to hold Excel data
    output = io.BytesIO()
    
    # Create Excel writer
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Create a summary sheet first
        summary_data = []
        for dataset_name, predictions in st.session_state['predictions'].items():
            if predictions:  # Skip empty prediction lists
                for pred in predictions:
                    pred_copy = pred.copy()
                    pred_copy['Dataset'] = dataset_name.replace('.csv', '')
                    summary_data.append(pred_copy)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='All Predictions', index=False)
        
        # Create individual sheets for each dataset
        for dataset_name, predictions in st.session_state['predictions'].items():
            if predictions:  # Skip empty prediction lists
                sheet_name = dataset_name.replace('.csv', '')
                if len(sheet_name) > 31:  # Excel sheet name length limit
                    sheet_name = sheet_name[:31]
                predictions_df = pd.DataFrame(predictions)
                predictions_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Reset pointer to start of buffer
    output.seek(0)
    
    # Create download button in the sidebar
    st.sidebar.download_button(
        label="📥 Download All Predictions",
        data=output,
        file_name="All_Predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def main():
    st.title('💲Cost Prediction RT2025💲')
    
    # Initialize session state variables
    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = {}
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = {}
    if 'processed_excel_files' not in st.session_state:
        st.session_state['processed_excel_files'] = set()
    
    # Sidebar: General Controls
    st.sidebar.header('Data Controls')
    if st.sidebar.button("Clear all predictions"):
        st.session_state['predictions'] = {}
        st.sidebar.success("All predictions cleared!")
    
    if st.sidebar.button("Clear processed files history"):
        st.session_state['processed_excel_files'] = set()
        st.sidebar.success("Processed files history cleared!")
    
    # Add Download All Predictions button in sidebar
    if st.sidebar.button("📥 Download All Predictions"):
        if st.session_state['predictions']:
            download_all_predictions()
            st.sidebar.success("All predictions compiled successfully!")
        else:
            st.sidebar.warning("No predictions to download.")
    
    # Upload CSV files
    with st.sidebar.expander("📂 Upload Data", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True
        )

    # Read and store datasets
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state['datasets']:
            df = pd.read_csv(uploaded_file)
            st.session_state['datasets'][uploaded_file.name] = df
            st.session_state['predictions'][uploaded_file.name] = []

    # Cleanup stale datasets
    uploaded_names = {f.name for f in uploaded_files}
    for name in list(st.session_state['datasets'].keys()):
        if name not in uploaded_names:
            del st.session_state['datasets'][name]
            st.session_state['predictions'].pop(name, None)

    if not st.session_state['datasets']:
        st.write("Please upload one or more CSV files to begin.")
        return

    selected_dataset_name = st.sidebar.selectbox(
        "Select a dataset for prediction",
        list(st.session_state['datasets'].keys())
    )
    df = st.session_state['datasets'][selected_dataset_name]
    clean_name = selected_dataset_name.replace('.csv', '')
    st.subheader(f"📊 Metrics: {clean_name}")

    # Get currency symbol from the dataset
    currency = get_currency_symbol(df)

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    st.header('Data Overview')
    st.write('Dataset Shape:', df_imputed.shape)
    st.dataframe(df_imputed.head())

    # Prepare features/target
    target_column = st.selectbox("Select the target variable to predict", df_imputed.columns)
    X = df_imputed.drop(columns=[target_column])
    y = df_imputed[target_column]

    st.header('Model Training')
    test_size = st.slider('Select test size (0.0-1.0)', 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)

    st.header('Model Performance')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    col1, col2 = st.columns(2)
    col1.metric('RMSE', f'{rmse:,.2f}')
    col2.metric('R² Score', f'{r2:.2f}')

    # Visualization Section - MODIFIED FOR HEIGHT-BASED DISPLAY
    with st.expander('Data Visualization', expanded=True):
        st.subheader('Correlation Matrix')
        
        # Calculate the optimal height based on feature count (more features = taller graph)
        feature_count = len(X.columns)
        corr_height = min(9, max(7, feature_count * 0.5))  # Adjust height proportionally to feature count with min/max limits
        
        # Create figure with height-optimized dimensions
        fig, ax = plt.subplots(figsize=(8, corr_height))
        sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 10})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader('Feature Importance')
        
        # Calculate optimal height for feature importance plot based on feature count
        fi_height = min(8, max(4, feature_count * 0.3))  # Height proportional to feature count with min/max limits
        
        fig, ax = plt.subplots(figsize=(8, fi_height))
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader('Cost Curve (Original Data Only)')
        feature = st.selectbox('Select feature for cost curve (Data Visualization)', X.columns, key='cost_curve_feature_viz')

        fig, ax = plt.subplots(figsize=(7, 6))
        x_vals = df_imputed[feature].values
        y_vals = y.values
        mask = (x_vals > 0) & (y_vals > 0)

        if mask.sum() >= 2:
            log_x = np.log(x_vals[mask])
            log_y = np.log(y_vals[mask])
            slope, intercept, r_val, _, _ = linregress(log_x, log_y)
            a = np.exp(intercept)
            b = slope
            sns.scatterplot(x=x_vals, y=y_vals, label='Original Data', ax=ax)
            x_line = np.linspace(min(x_vals[mask]), max(x_vals[mask]), 100)
            y_line = a * (x_line ** b)
            ax.plot(x_line, y_line, color='red', label=f'Fit: y = {a:.2f} * x^{b:.2f}')
            ax.text(0.05, 0.95, f'$R^2$ = {r_val**2:.3f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        else:
            sns.scatterplot(x=x_vals, y=y_vals, label='Original Data', ax=ax)
            st.warning("Not enough data for regression.")

        ax.set_xlabel(feature)
        ax.set_ylabel(target_column)
        ax.set_title(f'Cost Curve: {feature} vs {target_column}')
        ax.legend()

        # Apply human-readable format to axis labels
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.set_major_formatter(FuncFormatter(human_format))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.header('Make New Predictions')
    project_name = st.text_input('Enter Project Name')
    new_data = {}
    for col in X.columns:
        new_data[col] = st.number_input(f'{col}', value=float(X[col].mean()))

    if st.button('Predict'):
        df_input = pd.DataFrame([new_data])
        input_scaled = scaler.transform(df_input)
        pred = rf_model.predict(input_scaled)[0]
        result = {'Project Name': project_name, **new_data, target_column: round(pred, 2)}
        
        # Add EPCIC breakdown to result
        epcic_breakdown = {}
        for phase, percent in epcic_percentages.items():
            cost = round(pred * (percent / 100), 2)
            result[f"{phase} Cost"] = cost
            epcic_breakdown[phase] = {'cost': cost, 'percentage': percent}
        
        # Add PRR breakdown to result
        prr_breakdown = {}
        for phase, percent in prr_percentages.items():
            cost = round(pred * (percent / 100), 2)
            result[f"{phase} Cost"] = cost
            prr_breakdown[phase] = {'cost': cost, 'percentage': percent}
        
        # Add Pre-Dev and Owner's Cost breakdown to result
        predev_cost = round(pred * (predev_percentage / 100), 2)
        owners_cost = round(pred * (owners_percentage / 100), 2)
        result["Pre-Development Cost"] = predev_cost
        result["Owner's Cost"] = owners_cost
        
        # Add Cost Contingency (calculated on pred + owners_cost)
        contingency_base = pred + owners_cost
        contingency_cost = round(contingency_base * (contingency_percentage / 100), 2)
        result["Cost Contingency"] = contingency_cost
        
        # Add Escalation & Inflation (calculated on pred + owners_cost)
        escalation_base = pred + owners_cost
        escalation_cost = round(escalation_base * (escalation_percentage / 100), 2)
        result["Escalation & Inflation"] = escalation_cost
        
        # Calculate Grand Total 
        grand_total = round(pred + owners_cost + contingency_cost + escalation_cost, 2)
        result["Grand Total"] = grand_total
        
        st.session_state['predictions'][selected_dataset_name].append(result)
        
       # Build display text with all costs on separate lines
        display_text = f"### **✅Cost Summary of project {project_name}**\n\n**{target_column}:** {format_currency(pred, currency)}\n\n"
        
        # Check if any breakdown percentages are greater than 0
        has_breakdown = any(data['percentage'] > 0 for data in epcic_breakdown.values()) or \
                       any(data['percentage'] > 0 for data in prr_breakdown.values()) or \
                       predev_percentage > 0 or owners_percentage > 0 or \
                       contingency_percentage > 0 or escalation_percentage > 0
        
        if has_breakdown:
            # Add EPCIC breakdown (only if percentage > 0)
            for phase, data in epcic_breakdown.items():
                if data['percentage'] > 0:
                    display_text += f"• {phase} ({data['percentage']:.1f}%): {format_currency(data['cost'], currency)}\n\n"
            
            # Add PRR breakdown (only if percentage > 0)  
            for phase, data in prr_breakdown.items():
                if data['percentage'] > 0:
                    display_text += f"• {phase} ({data['percentage']:.2f}%): {format_currency(data['cost'], currency)}\n\n"
            
            # Add other cost items (each on separate line, only if percentage > 0)
            if predev_percentage > 0:
                display_text += f"**Pre-Development ({predev_percentage:.1f}%):** {format_currency(predev_cost, currency)}\n\n"
            
            if owners_percentage > 0:
                display_text += f"**Owner's Cost ({owners_percentage:.1f}%):** {format_currency(owners_cost, currency)}\n\n"
            
            if contingency_percentage > 0:
                display_text += f"**Contingency ({contingency_percentage:.1f}%):** {format_currency(contingency_cost, currency)}\n\n"
            
            if escalation_percentage > 0:
                display_text += f"**Escalation & Inflation ({escalation_percentage:.3f}%):** {format_currency(escalation_cost, currency)}\n\n"
        
        display_text += f"**Grand Total (Predicted Cost + Owner's Cost + Contingency + Esc Infl):** {format_currency(grand_total, currency)}"
        
        st.success(display_text)
        
    st.write("Or upload an Excel file:")
    excel_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if excel_file:
        file_id = f"{excel_file.name}_{excel_file.size}_{selected_dataset_name}"
        if file_id not in st.session_state['processed_excel_files']:
            batch_df = pd.read_excel(excel_file)
            if set(X.columns).issubset(batch_df.columns):
                scaled = scaler.transform(batch_df[X.columns])
                preds = rf_model.predict(scaled)
                batch_df[target_column] = preds
                for i, row in batch_df.iterrows():
                    name = row.get("Project Name", f"Project {i+1}")
                    entry = {'Project Name': name}
                    entry.update(row[X.columns].to_dict())
                    entry[target_column] = round(preds[i], 2)
                    
                    # Add EPCIC breakdown for batch predictions
                    for phase, percent in epcic_percentages.items():
                        cost = round(preds[i] * (percent / 100), 2)
                        entry[f"{phase} Cost"] = cost

                    # Add PRR breakdown for batch predictions
                    for phase, percent in prr_percentages.items():
                        cost = round(preds[i] * (percent / 100), 2)
                        entry[f"{phase} Cost"] = cost
                    
                    # Add Pre-Dev and Owner's Cost breakdown for batch predictions
                    predev_cost = round(preds[i] * (predev_percentage / 100), 2)
                    owners_cost = round(preds[i] * (owners_percentage / 100), 2)
                    entry["Pre-Development Cost"] = predev_cost
                    entry["Owner's Cost"] = owners_cost

                    # Add Cost Contingency to batch predictions
                    contingency_base = preds[i] + owners_cost
                    contingency_cost = round(contingency_base * (contingency_percentage / 100), 2)
                    entry["Cost Contingency"] = contingency_cost

                    # Add Escalation & Inflation to batch predictions
                    escalation_base = preds[i] + owners_cost
                    escalation_cost = round(escalation_base * (escalation_percentage / 100), 2)
                    entry["Escalation & Inflation"] = escalation_cost

                    # Calculate Grand Total for batch predictions
                    grand_total = round(preds[i] + owners_cost + contingency_cost + escalation_cost, 2)
                    entry["Grand Total"] = grand_total
                    
                    st.session_state['predictions'][selected_dataset_name].append(entry)
                st.session_state['processed_excel_files'].add(file_id)
                st.success("Batch prediction successful!")
            else:
                st.error("Excel missing required columns.")

    # Updated Cost Curve section with predictions overlay
    st.subheader('📈 Cost Curve with Predictions')
    feature = st.selectbox('Cost Curve Dropdown Menu', X.columns, key='cost_curve_feature')

    fig, ax = plt.subplots(figsize=(7, 6))

    # Original data
    x_vals = df_imputed[feature].values
    y_vals = y.values
    mask = (x_vals > 0) & (y_vals > 0)

    if mask.sum() >= 2:
        log_x = np.log(x_vals[mask])
        log_y = np.log(y_vals[mask])
        slope, intercept, r_val, _, _ = linregress(log_x, log_y)
        a = np.exp(intercept)
        b = slope
        sns.scatterplot(x=x_vals, y=y_vals, label='Original Data', ax=ax)
        x_line = np.linspace(min(x_vals[mask]), max(x_vals[mask]), 100)
        y_line = a * (x_line ** b)
        ax.plot(x_line, y_line, color='red', label=f'Fit: y = {a:.2f} * x^{b:.2f}')
        ax.text(0.05, 0.95, f'$R^2$ = {r_val**2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    else:
        sns.scatterplot(x=x_vals, y=y_vals, label='Original Data', ax=ax)
        st.warning("Not enough data for regression.")

    # Predictions overlay
    prediction_data = st.session_state['predictions'].get(selected_dataset_name, [])
    if prediction_data:
        df_preds = pd.DataFrame(prediction_data)
        if feature in df_preds.columns and target_column in df_preds.columns:
            sns.scatterplot(
                data=df_preds,
                x=feature,
                y=target_column,
                marker='X',
                s=100,
                color='green',
                label='Predictions',
                ax=ax
            )

    ax.set_xlabel(feature)
    ax.set_ylabel(target_column)
    ax.set_title(f'Cost Curve: {feature} vs {target_column}')
    ax.legend()

    # Apply human-readable format to axis labels
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Project list & delete buttons
    with st.expander('Simplified Project List', expanded=True):
        preds = st.session_state['predictions'][selected_dataset_name]
        if preds:
            if st.button('Delete All', key='delete_all'):
                st.session_state['predictions'][selected_dataset_name] = []
                to_remove = {fid for fid in st.session_state['processed_excel_files'] if fid.endswith(selected_dataset_name)}
                for fid in to_remove:
                    st.session_state['processed_excel_files'].remove(fid)
                st.rerun()

            for i, p in enumerate(preds):
                c1, c2 = st.columns([3, 1])
                c1.write(p['Project Name'])
                if c2.button('Delete', key=f'del_{i}'):
                    preds.pop(i)
                    st.rerun()
        else:
            st.write("No predictions yet.")

    st.header(f"Prediction Summary based on {clean_name}")
    if preds := st.session_state['predictions'][selected_dataset_name]:
        df_preds = pd.DataFrame(preds)
        
        # Format numeric columns for display with commas
        num_cols = df_preds.select_dtypes(include=[np.number]).columns
        df_preds_display = df_preds.copy()
        for col in num_cols:
            df_preds_display[col] = df_preds_display[col].apply(lambda x: format_with_commas(x))
        
        st.dataframe(df_preds_display, use_container_width=True)
        towrite = io.BytesIO()
        df_preds.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(
            "Download Predictions as Excel",
            data=towrite,
            file_name=f"{selected_dataset_name}_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.write("No predictions available.")

# Entry point
if __name__ == '__main__':
    # First check if the user is authenticated with just a password
    if check_password():
        main()
    else:
        st.error("Please enter the correct password to access the Cost Prediction Tool")
