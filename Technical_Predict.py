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
from matplotlib.ticker import FuncFormatter

st.set_page_config(
    page_title="General Predictor 2025",
    page_icon="⭐",
    initial_sidebar_state="expanded"
)

def human_format(num, pos=None):
    if num >= 1e9:
        return f'{num/1e9:.1f}B'
    elif num >= 1e6:
        return f'{num/1e6:.1f}M'
    elif num >= 1e3:
        return f'{num/1e3:.1f}K'
    else:
        return f'{num:.0f}'

def format_with_commas(num):
    return f"{num:,.2f}"

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" in st.session_state:
        return st.session_state["password_correct"]

    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=password_entered)
    return False

def main():
    st.title('📈General Predictor Tool 2025📈')

    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = {}
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = {}
    if 'processed_excel_files' not in st.session_state:
        st.session_state['processed_excel_files'] = set()

    st.sidebar.header('Data Controls')
    if st.sidebar.button("Clear all predictions"):
        st.session_state['predictions'] = {}
        st.sidebar.success("All predictions cleared!")

    with st.sidebar.expander("📂 Upload Data", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True
        )

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state['datasets']:
            df = pd.read_csv(uploaded_file)
            st.session_state['datasets'][uploaded_file.name] = df
            st.session_state['predictions'][uploaded_file.name] = []

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

    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    st.header('Data Overview')
    st.write('Dataset Shape:', df_imputed.shape)
    st.dataframe(df_imputed.head())

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

    with st.expander('Data Visualization', expanded=True):
        st.subheader('Correlation Matrix')
        fig, ax = plt.subplots(figsize=(8, min(9, max(7, len(X.columns) * 0.5))))
        sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 10})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader('Feature Importance')
        fig, ax = plt.subplots(figsize=(8, min(8, max(4, len(X.columns) * 0.3))))
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader(f'Prediction Curve (Target: {target_column})')
        feature = st.selectbox('Select feature for prediction curve (Data Visualization)', X.columns, key='curve_feature_viz')
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
        ax.set_title(f'Prediction Curve: {feature} vs {target_column}')
        ax.legend()
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.set_major_formatter(FuncFormatter(human_format))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Updated Cost Curve section with predictions overlay
    st.subheader('📈 Cost Curve with Predictions')
    feature = st.selectbox('Cost Curve Dropdown Menu', X.columns, key='cost_curve_feature')

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
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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

    st.header(f"Prediction Summary based on {clean_name} (Target: {target_column})")
    preds = st.session_state['predictions'][selected_dataset_name]
    if preds:
        df_preds = pd.DataFrame(preds)
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

if __name__ == '__main__':
    if check_password():
        main()
    else:
        st.error("Please enter the correct password to access the General Predictor Tool 2025")
