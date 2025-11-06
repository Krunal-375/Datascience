import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Cold Chain Subsidy Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2c3e50;
    margin: 1rem 0;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # --- FIX APPLIED HERE: Replaced relative paths with the user's absolute paths ---
    
    # Define absolute paths for the datasets
    subsidy_path = '/Users/krunal/Documents/Sem7/DataScience/cleaned_subsidy_dataset.csv'
    cost_path = '/Users/krunal/Documents/Sem7/DataScience/Integrated Cold Chain Cost Report.csv'
    
    # Load cleaned subsidy data
    try:
        df_subsidy = pd.read_csv(subsidy_path)
    except FileNotFoundError:
        st.error(f"Error: The cleaned subsidy dataset was not found at {subsidy_path}. Please check the path.")
        return None, None
        
    # Load integrated cost report
    try:
        df_cost = pd.read_csv(cost_path)
    except FileNotFoundError:
        st.error(f"Error: The integrated cost report was not found at {cost_path}. Please check the path.")
        return None, None

    # Data Preprocessing for df_subsidy
    # Convert 'project_cost' and 'amount_sanct' to numeric, handling errors
    for col in ['project_cost', 'amount_sanct']:
        df_subsidy[col] = pd.to_numeric(df_subsidy[col], errors='coerce')
    
    # Clean 'sanction_year'
    df_subsidy['sanction_year'] = df_subsidy['sanction_year'].astype('Int64')
    df_subsidy.dropna(subset=['sanction_year'], inplace=True)
    
    # Impute missing 'subsidy_ratio' with the median for the overall df, then category-wise
    median_ratio = df_subsidy['subsidy_ratio'].median()
    df_subsidy['subsidy_ratio'].fillna(median_ratio, inplace=True)
    
    # Clean up state and district names
    df_subsidy['state_name'] = df_subsidy['state_name'].str.strip()
    df_subsidy['district_name'] = df_subsidy['district_name'].str.strip()
    
    # Add project_size_category if it was missing (based on original logic)
    if 'project_size_category' not in df_subsidy.columns:
        bins = [0, 100, 500, 1000, np.inf]
        labels = ['Small (<₹1 Cr)', 'Medium (₹1-5 Cr)', 'Large (₹5-10 Cr)', 'Very Large (>₹10 Cr)']
        df_subsidy['project_size_category'] = pd.cut(df_subsidy['project_cost'] / 100, bins=bins, labels=labels, right=False)

    # Data Preprocessing for df_cost
    # Clean column names
    df_cost.columns = df_cost.columns.str.lower().str.replace('[^a-zA-Z0-9_]', '', regex=True)
    df_cost.rename(columns={'id': 'id_cost'}, inplace=True)

    return df_subsidy, df_cost

df_subsidy, df_cost = load_data()

# Check if data loading failed
if df_subsidy is None or df_cost is None:
    st.stop()


# --- Main Dashboard Structure ---

st.markdown("<h1 class='main-header'>Cold Chain Subsidy Analysis</h1>", unsafe_allow_html=True)
st.markdown("A deep dive into project funding, cost analysis, and subsidy ratio effectiveness across India.", unsafe_allow_html=True)
st.markdown("---")


# Sidebar for Navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["Overview & Metrics", "Geographical Analysis", "Cost & Project Deep Dive", "Predictive Modeling"])

# --- Overview & Metrics Page ---
if page == "Overview & Metrics":
    
    st.header("Overall Performance Metrics")
    
    # Calculate key metrics
    total_projects = df_subsidy.shape[0]
    total_project_cost = df_subsidy['project_cost'].sum()
    total_amount_sanct = df_subsidy['amount_sanct'].sum()
    
    # Calculate average subsidy ratio excluding cases where cost is 0
    df_valid_ratio = df_subsidy[df_subsidy['project_cost'] > 0]
    avg_subsidy_ratio = (df_valid_ratio['amount_sanct'].sum() / df_valid_ratio['project_cost'].sum()) if df_valid_ratio['project_cost'].sum() > 0 else 0
    
    # Calculate number of unique states and districts
    unique_states = df_subsidy['state_name'].nunique()
    unique_districts = df_subsidy['district_name'].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Projects", f"{total_projects:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Project Cost (Cr)", f"₹{total_project_cost:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Subsidy Sanctioned (Cr)", f"₹{total_amount_sanct:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. Subsidy Ratio", f"{avg_subsidy_ratio:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col5:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("States/Districts Covered", f"{unique_states} / {unique_districts}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Time Series Analysis
    st.markdown("<h2 class='sub-header'>Subsidy and Cost Trends Over Time</h2>", unsafe_allow_html=True)
    
    # Group by sanction_year
    df_time = df_subsidy.groupby('sanction_year').agg(
        Total_Cost=('project_cost', 'sum'),
        Total_Subsidy=('amount_sanct', 'sum'),
        Project_Count=('id', 'count')
    ).reset_index().sort_values('sanction_year')
    
    # Ensure all years are present for smooth line
    min_year = df_time['sanction_year'].min()
    max_year = df_time['sanction_year'].max()
    all_years = pd.DataFrame({'sanction_year': range(min_year, max_year + 1)})
    df_time = pd.merge(all_years, df_time, on='sanction_year', how='left').fillna(0)
    
    # Dual Axis Plot
    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar for Project Count
    fig_time.add_trace(go.Bar(x=df_time['sanction_year'], y=df_time['Project_Count'], name='Project Count', marker_color='#2ca02c'), secondary_y=False)
    
    # Line for Total Subsidy
    fig_time.add_trace(go.Scatter(x=df_time['sanction_year'], y=df_time['Total_Subsidy'], name='Total Subsidy (Cr)', mode='lines+markers', line=dict(color='#ff7f0e')), secondary_y=True)
    
    fig_time.update_layout(
        title_text='Annual Project Count and Subsidy Sanctioned (Cr)',
        xaxis_title="Sanction Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_time.update_yaxes(title_text="Project Count", secondary_y=False)
    fig_time.update_yaxes(title_text="Total Subsidy (Cr)", secondary_y=True)
    
    st.plotly_chart(fig_time, use_container_width=True)

    # Project Size Distribution
    st.markdown("<h2 class='sub-header'>Project Size Distribution</h2>", unsafe_allow_html=True)
    df_size = df_subsidy.groupby('project_size_category', observed=True)['id'].count().reset_index(name='Count')
    fig_size = px.pie(df_size, values='Count', names='project_size_category', title='Distribution by Project Cost Category',
                      hole=0.3, color_discrete_sequence=px.colors.sequential.Agsunset)
    st.plotly_chart(fig_size, use_container_width=True)

# --- Geographical Analysis Page ---
elif page == "Geographical Analysis":
    
    st.header("Geographical Distribution of Projects")
    
    # Top 10 States by Project Count
    st.markdown("<h2 class='sub-header'>Top 10 States by Project Count</h2>", unsafe_allow_html=True)
    df_state_count = df_subsidy.groupby('state_name')['id'].count().reset_index(name='Project Count').sort_values('Project Count', ascending=False).head(10)
    fig_state_count = px.bar(df_state_count, x='Project Count', y='state_name', orientation='h', 
                             title='Top 10 States by Number of Projects', color='Project Count',
                             color_continuous_scale=px.colors.sequential.Blues_r)
    fig_state_count.update_yaxes(title_text="State Name", categoryorder='total ascending')
    st.plotly_chart(fig_state_count, use_container_width=True)
    
    st.markdown("---")

    # Top 10 States by Subsidy Amount
    st.markdown("<h2 class='sub-header'>Top 10 States by Total Subsidy Sanctioned (Cr)</h2>", unsafe_allow_html=True)
    df_state_subsidy = df_subsidy.groupby('state_name')['amount_sanct'].sum().reset_index(name='Total Subsidy (Cr)').sort_values('Total Subsidy (Cr)', ascending=False).head(10)
    fig_state_subsidy = px.bar(df_state_subsidy, x='Total Subsidy (Cr)', y='state_name', orientation='h', 
                                title='Top 10 States by Total Subsidy (Cr)', color='Total Subsidy (Cr)',
                                color_continuous_scale=px.colors.sequential.Reds_r)
    fig_state_subsidy.update_yaxes(title_text="State Name", categoryorder='total ascending')
    st.plotly_chart(fig_state_subsidy, use_container_width=True)

    st.markdown("---")

    # District Level Analysis
    st.markdown("<h2 class='sub-header'>District Level Subsidy Analysis</h2>", unsafe_allow_html=True)
    
    state_list = df_subsidy['state_name'].unique().tolist()
    selected_state = st.selectbox("Select State for District Analysis", state_list)
    
    df_district = df_subsidy[df_subsidy['state_name'] == selected_state]
    df_district_summary = df_district.groupby('district_name').agg(
        Total_Subsidy=('amount_sanct', 'sum'),
        Project_Count=('id', 'count')
    ).reset_index().sort_values('Total_Subsidy', ascending=False).head(10)
    
    if not df_district_summary.empty:
        fig_district = px.bar(df_district_summary, x='Total_Subsidy', y='district_name', orientation='h',
                              title=f'Top Districts in {selected_state} by Total Subsidy (Cr)',
                              hover_data=['Project_Count'], color='Project_Count',
                              color_continuous_scale=px.colors.sequential.Viridis)
        fig_district.update_yaxes(title_text="District Name", categoryorder='total ascending')
        st.plotly_chart(fig_district, use_container_width=True)
    else:
        st.info(f"No district data available for {selected_state}.")

# --- Cost & Project Deep Dive Page ---
elif page == "Cost & Project Deep Dive":
    
    st.header("Project Cost & Subsidy Ratios")
    
    # Relationship between Cost and Subsidy
    st.markdown("<h2 class='sub-header'>Project Cost vs. Subsidy Sanctioned</h2>", unsafe_allow_html=True)
    
    # Filter out extreme outliers for better visualization
    df_scatter = df_subsidy[(df_subsidy['project_cost'] < 2000) & (df_subsidy['amount_sanct'] < 1000)].copy()
    
    fig_scatter = px.scatter(df_scatter, x='project_cost', y='amount_sanct', 
                             color='project_size_category', 
                             hover_data=['state_name', 'district_name'],
                             title='Project Cost vs. Subsidy Sanctioned (Crores)',
                             labels={'project_cost': 'Project Cost (Cr)', 'amount_sanct': 'Subsidy Sanctioned (Cr)'})
    
    # Add a diagonal line for reference (e.g., a 50% subsidy line)
    max_val = max(df_scatter['project_cost'].max(), df_scatter['amount_sanct'].max())
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val * 0.5], 
                                     mode='lines', name='50% Subsidy', 
                                     line=dict(dash='dash', color='red')))
    
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Subsidy Ratio Distribution
    st.markdown("<h2 class='sub-header'>Subsidy Ratio Distribution</h2>", unsafe_allow_html=True)
    
    # Filter for valid ratios (0 to 1) for better histogram visualization
    df_valid_ratio = df_subsidy[(df_subsidy['subsidy_ratio'] >= 0) & (df_subsidy['subsidy_ratio'] <= 1)]
    
    fig_hist = px.histogram(df_valid_ratio, x='subsidy_ratio', nbins=30, 
                            title='Distribution of Subsidy Ratios (0 to 1)',
                            labels={'subsidy_ratio': 'Subsidy Ratio (Sanctioned / Cost)'},
                            color_discrete_sequence=['#1f77b4'])
    
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    
    # Integrated Cost Report Analysis (K-Means Clustering)
    st.markdown("<h2 class='sub-header'>Integrated Cold Chain Project Cost Clustering</h2>", unsafe_allow_html=True)
    st.info("Applying K-Means Clustering on cost report data to identify natural groupings of project costs.")
    
    # Data Cleaning and Preparation for Clustering
    # Identify numeric columns for clustering (excluding IDs and non-cost columns)
    cost_cols = df_cost.select_dtypes(include=np.number).columns.tolist()
    # Filter out columns that are primarily identifiers or non-cost
    exclude_cols = ['id_cost', 'state_code', 'district_code', 'year_of_subsidy_sanct'] 
    cluster_cols = [col for col in cost_cols if col not in exclude_cols and df_cost[col].isnull().sum() < len(df_cost) * 0.5]

    df_cluster = df_cost[cluster_cols].dropna().copy()
    
    if len(df_cluster) > 100 and len(cluster_cols) >= 2:
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)
        
        # Determine optimal number of clusters (Elbow Method)
        inertia = []
        K_range = range(2, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        # Use a fixed number of clusters (e.g., 3) for display simplicity
        best_k = 3
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
        df_cluster['Cluster'] = df_cluster['Cluster'].astype(str)

        # Plot 2 most variance-explaining features, if available
        feature1 = cluster_cols[0]
        feature2 = cluster_cols[1]
        
        st.markdown(f"**Clustering on:** {', '.join(cluster_cols)}")

        fig_cluster = px.scatter(df_cluster, x=feature1, y=feature2, color='Cluster',
                                 title=f'K-Means Clustering ({best_k} Clusters)',
                                 color_discrete_sequence=px.colors.qualitative.Dark24)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Display cluster statistics
        cluster_summary = df_cluster.groupby('Cluster')[cluster_cols].mean().T
        st.subheader("Cluster Mean Values")
        st.dataframe(cluster_summary.style.highlight_max(axis=1))
        
    else:
        st.warning(f"Not enough data points ({len(df_cluster)}) or features ({len(cluster_cols)}) in the cost report for meaningful clustering analysis.")

# --- Predictive Modeling Page ---
elif page == "Predictive Modeling":
    
    st.header("Subsidy Amount Prediction (Multiple Linear Regression)")
    st.info("Predicting the sanctioned subsidy amount based on project cost and other categorical features.")
    
    # Feature Engineering and Preparation
    df_model = df_subsidy.copy()
    
    # Drop rows where 'project_cost' or 'amount_sanct' is 0 or NaN for meaningful regression
    df_model.dropna(subset=['project_cost', 'amount_sanct', 'sanction_year'], inplace=True)
    df_model = df_model[(df_model['project_cost'] > 0) & (df_model['amount_sanct'] > 0)]

    # Select features (numerical + categorical)
    numerical_features = ['project_cost', 'sanction_year']
    categorical_features = ['state_name', 'project_size_category']
    
    # Check if a sufficient number of numerical features exists
    if len(numerical_features) == 0:
        st.error("No valid numerical features found for modeling.")
    else:
        # Convert categorical features to one-hot encoded variables
        df_model_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True, dtype=int)
        
        # Define features (X) and target (y)
        # We drop all original columns that are now one-hot encoded, plus the target, ratio, and non-predictive columns.
        X_cols = [col for col in df_model_encoded.columns if col not in ['amount_sanct', 'subsidy_ratio', 'id', 'project_code', 'year_of_subsidy_sanct', 'district_name', 'state_code', 'district_code', 'agency', 'supported_by', 'beneficiary_name', 'project_address', 'current_status', 'state_name', 'project_size_category']]
        
        X = df_model_encoded[X_cols]
        y = df_model_encoded['amount_sanct']
        
        # Check for sufficient data
        if X.shape[0] > 50 and X.shape[1] > 0:
            
            # Train-Test Split
            X_train, X_test, y_train_mlr, y_test_mlr = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Standardize numerical features
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
            X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

            # Multiple Linear Regression Model
            model_mlr = LinearRegression()
            model_mlr.fit(X_train_scaled, y_train_mlr)
            
            # Prediction
            y_pred_test_mlr = model_mlr.predict(X_test_scaled)
            
            # Evaluation
            r2 = r2_score(y_test_mlr, y_pred_test_mlr)
            rmse = np.sqrt(mean_squared_error(y_test_mlr, y_pred_test_mlr))

            st.subheader("Model Performance")
            col_r2, col_rmse = st.columns(2)
            with col_r2:
                st.metric("R-squared (R²)", f"{r2:.4f}")
            with col_rmse:
                st.metric("Root Mean Squared Error (RMSE)", f"₹{rmse:.2f} Cr")
            
            st.markdown("---")
            
            st.subheader("Model Insights")

            # Feature Importance (Coefficients)
            importance_df = pd.DataFrame({
                'Feature': X_train_scaled.columns,
                'Coefficient': model_mlr.coef_
            }).sort_values('Coefficient', key=abs, ascending=False).head(15) # Show top 15 features for clarity
            
            fig_importance = px.bar(importance_df, x='Coefficient', y='Feature',
                                      orientation='h', title="Top 15 Feature Importance (Standardized Coeff.)")
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Actual vs Predicted
            fig_pred = px.scatter(x=y_test_mlr, y=y_pred_test_mlr,
                                title="Actual vs Predicted Subsidy Sanctioned Values",
                                labels={'x': 'Actual Subsidy (Cr)', 'y': 'Predicted Subsidy (Cr)'},
                                trendline="ols")
            
            # Add perfect prediction line
            min_val = min(y_test_mlr.min(), y_pred_test_mlr.min())
            max_val = max(y_test_mlr.max(), y_pred_test_mlr.max())
            fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect Prediction',
                                        line=dict(color='red', dash='dash')))
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning(f"Not enough data points ({X.shape[0]}) or features ({X.shape[1]}) for multiple linear regression analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    📊 Cold Chain Subsidy Dashboard | Built with Streamlit & Plotly
    </div>
    """, unsafe_allow_html=True)
