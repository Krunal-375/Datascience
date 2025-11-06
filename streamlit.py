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
    try:
        df = pd.read_csv('cleaned_subsidy_dataset.csv')
        
        # Debug: Print original columns
        print(f"Original columns: {df.columns.tolist()}")
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate columns if any exist (keep first occurrence)
        if df.columns.duplicated().any():
            print(f"Found duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            print(f"After removing duplicates: {df.columns.tolist()}")
        
        # Clean column names - remove any whitespace and special characters
        df.columns = df.columns.str.strip()
        
        # Ensure column names are unique by adding suffix to duplicates
        cols = df.columns.tolist()
        seen = set()
        new_cols = []
        for col in cols:
            if col in seen:
                counter = 1
                new_col = f"{col}_{counter}"
                while new_col in seen:
                    counter += 1
                    new_col = f"{col}_{counter}"
                new_cols.append(new_col)
                seen.add(new_col)
            else:
                new_cols.append(col)
                seen.add(col)
        
        df.columns = new_cols
        
        # Convert numeric columns
        numeric_cols = ['project_cost', 'amount_sanct', 'sanction_year', 'subsidy_ratio']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Final columns: {df.columns.tolist()}")
        print(f"Final shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error in load_data: {e}")
        raise e

# Main title
st.markdown('<h1 class="main-header">🏭 Cold Chain Subsidy Dashboard</h1>', unsafe_allow_html=True)

# Load data
try:
    # Clear cache to ensure fresh data loading
    st.cache_data.clear()
    df = load_data()
    st.success(f"Data loaded successfully! Total records: {len(df)}")
    
    # Define column types for all analyses
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Debug: Show column info
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("Columns:", df.columns.tolist())
        st.sidebar.write("Shape:", df.shape)
        duplicates = df.columns[df.columns.duplicated()].tolist()
        st.sidebar.write("Duplicate columns:", duplicates)
        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", 
     "Top 10 Analysis", "Distribution Analysis", "Clustering Analysis", "ML Models"]
)

# Overview Section
if analysis_type == "Overview":
    st.markdown('<h2 class="sub-header">📈 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_projects = len(df)
        st.metric("Total Projects", f"{total_projects:,}")
    
    with col2:
        total_amount = df['amount_sanct'].sum() / 1000  # Convert to thousands
        st.metric("Total Sanctioned Amount", f"₹{total_amount:,.0f}K")
    
    with col3:
        avg_subsidy_ratio = df['subsidy_ratio'].mean()
        st.metric("Avg Subsidy Ratio", f"{avg_subsidy_ratio:.2%}")
    
    with col4:
        unique_states = df['state_name'].nunique()
        st.metric("States Covered", unique_states)
    
    # Data preview
    st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    # Basic statistics
    st.markdown('<h3 class="sub-header">Basic Statistics</h3>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

# Univariate Analysis
elif analysis_type == "Univariate Analysis":
    st.markdown('<h2 class="sub-header">📊 Univariate Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric variable analysis
        st.markdown("### Numeric Variables")
        selected_numeric = st.selectbox("Select Numeric Variable", numeric_columns)
        
        if selected_numeric:
            fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot
            fig_box = px.box(df, y=selected_numeric, title=f"Box Plot of {selected_numeric}")
            st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Categorical variable analysis
        st.markdown("### Categorical Variables")
        selected_categorical = st.selectbox("Select Categorical Variable", categorical_columns)
        
        if selected_categorical:
            value_counts = df[selected_categorical].value_counts().head(10)
            fig = px.bar(x=value_counts.values, y=value_counts.index, 
                        orientation='h', title=f"Top 10 {selected_categorical}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie chart
            fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
                            title=f"Distribution of {selected_categorical}")
            st.plotly_chart(fig_pie, use_container_width=True)

# Bivariate Analysis
elif analysis_type == "Bivariate Analysis":
    st.markdown('<h2 class="sub-header">🔍 Bivariate Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X Variable", numeric_columns + categorical_columns)
    with col2:
        y_var = st.selectbox("Select Y Variable", numeric_columns)
    
    if x_var and y_var:
        try:
            # Create a completely clean dataframe for plotting
            # Only select the specific columns we need and ensure no duplicates
            clean_df = df.copy()
            
            # Ensure no duplicate columns in the working dataframe
            if clean_df.columns.duplicated().any():
                clean_df = clean_df.loc[:, ~clean_df.columns.duplicated(keep='first')]
            
            # Create plot data with only the needed columns
            if x_var in clean_df.columns and y_var in clean_df.columns:
                plot_data = clean_df[[x_var, y_var]].copy().dropna()
                
                # Reset index to avoid any index-related issues
                plot_data = plot_data.reset_index(drop=True)
                
                if len(plot_data) > 0:
                    if x_var in numeric_columns:
                        # Scatter plot for numeric vs numeric
                        fig = px.scatter(plot_data, x=x_var, y=y_var, 
                                       title=f"{y_var} vs {x_var}",
                                       trendline="ols")
                        st.plotly_chart(fig, width='stretch')
                        
                        # Correlation
                        correlation = plot_data[x_var].corr(plot_data[y_var])
                        st.metric("Correlation Coefficient", f"{correlation:.3f}")
                    else:
                        # Box plot for categorical vs numeric
                        fig = px.box(plot_data, x=x_var, y=y_var, title=f"{y_var} by {x_var}")
                        st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("No valid data available for the selected variables.")
            else:
                st.error(f"Selected columns not found in dataframe. Available columns: {list(clean_df.columns)}")
                
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            st.write("Debug info:")
            st.write(f"DataFrame columns: {df.columns.tolist()}")
            st.write(f"DataFrame shape: {df.shape}")
            st.write(f"Selected X variable: {x_var}")
            st.write(f"Selected Y variable: {y_var}")
    
    # Correlation matrix
    st.markdown("### Correlation Matrix")
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            # Remove duplicate columns if any
            numeric_df = numeric_df.loc[:, ~numeric_df.columns.duplicated(keep='first')]
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for correlation matrix.")
    except Exception as e:
        st.error(f"Error creating correlation matrix: {str(e)}")

# Multivariate Analysis
elif analysis_type == "Multivariate Analysis":
    st.markdown('<h2 class="sub-header">🎯 Multivariate Analysis</h2>', unsafe_allow_html=True)
    
    # 3D Scatter Plot
    st.markdown("### 3D Scatter Plot")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_3d = st.selectbox("X Axis", numeric_columns, key="3d_x")
    with col2:
        y_3d = st.selectbox("Y Axis", numeric_columns, key="3d_y")
    with col3:
        z_3d = st.selectbox("Z Axis", numeric_columns, key="3d_z")
    
    if x_3d and y_3d and z_3d:
        fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d,
                           color='current_status' if 'current_status' in df.columns else None,
                           title=f"3D Plot: {x_3d} vs {y_3d} vs {z_3d}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Parallel coordinates
    st.markdown("### Parallel Coordinates")
    selected_features = st.multiselect("Select Features for Parallel Coordinates", 
                                     numeric_columns, default=numeric_columns[:4])
    
    if len(selected_features) > 1:
        fig = px.parallel_coordinates(df, dimensions=selected_features,
                                    color='subsidy_ratio' if 'subsidy_ratio' in df.columns else None,
                                    title="Parallel Coordinates Plot")
        st.plotly_chart(fig, use_container_width=True)

# Top 10 Analysis
elif analysis_type == "Top 10 Analysis":
    st.markdown('<h2 class="sub-header">🏆 Top 10 Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Top Districts", "Top Projects", "Top States", "Top Agencies"])
    
    with tab1:
        st.markdown("### Top 10 Districts by Sanctioned Amount")
        top_districts = df.groupby('district_name')['amount_sanct'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_districts.values, y=top_districts.index, orientation='h',
                    title="Top 10 Districts by Total Sanctioned Amount")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top districts by project count
        st.markdown("### Top 10 Districts by Project Count")
        top_districts_count = df['district_name'].value_counts().head(10)
        fig2 = px.bar(x=top_districts_count.values, y=top_districts_count.index, orientation='h',
                     title="Top 10 Districts by Number of Projects")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown("### Top 10 Projects by Sanctioned Amount")
        top_projects = df.nlargest(10, 'amount_sanct')[['project_code', 'beneficiary_name', 'amount_sanct', 'state_name']]
        st.dataframe(top_projects, use_container_width=True)
        
        fig = px.bar(top_projects, x='amount_sanct', y='project_code',
                    orientation='h', title="Top 10 Projects by Sanctioned Amount")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Top 10 States by Sanctioned Amount")
        top_states = df.groupby('state_name')['amount_sanct'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_states.values, y=top_states.index, orientation='h',
                    title="Top 10 States by Total Sanctioned Amount")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Top 10 Agencies by Project Count")
        top_agencies = df['agency'].value_counts().head(10)
        fig = px.pie(values=top_agencies.values, names=top_agencies.index,
                    title="Top 10 Agencies by Number of Projects")
        st.plotly_chart(fig, use_container_width=True)

# Distribution Analysis
elif analysis_type == "Distribution Analysis":
    st.markdown('<h2 class="sub-header">📈 Distribution Analysis</h2>', unsafe_allow_html=True)
    
    # Select variable for distribution analysis
    selected_var = st.selectbox("Select Variable for Distribution Analysis", numeric_columns)
    
    if selected_var:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with normal curve
            fig = ff.create_distplot([df[selected_var].dropna()], [selected_var], 
                                   bin_size=None, show_hist=True, show_rug=False)
            fig.update_layout(title=f"Distribution of {selected_var}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate skewness
            skewness = df[selected_var].skew()
            if skewness > 0:
                skew_type = "Right Skewed (Positive)"
            elif skewness < 0:
                skew_type = "Left Skewed (Negative)"
            else:
                skew_type = "Normal Distribution"
            
            st.metric("Skewness", f"{skewness:.3f}")
            st.info(f"Distribution Type: {skew_type}")
        
        with col2:
            # Q-Q plot
            from scipy import stats
            
            # Generate Q-Q plot data
            sorted_data = np.sort(df[selected_var].dropna())
            n = len(sorted_data)
            theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
            
            fig = px.scatter(x=theoretical_quantiles, y=sorted_data,
                           title=f"Q-Q Plot for {selected_var}")
            fig.add_trace(go.Scatter(x=theoretical_quantiles, 
                                   y=theoretical_quantiles * np.std(sorted_data) + np.mean(sorted_data),
                                   mode='lines', name='Normal Line'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("### Statistical Summary")
            stats_summary = {
                "Mean": df[selected_var].mean(),
                "Median": df[selected_var].median(),
                "Standard Deviation": df[selected_var].std(),
                "Variance": df[selected_var].var(),
                "Min": df[selected_var].min(),
                "Max": df[selected_var].max(),
                "Skewness": df[selected_var].skew(),
                "Kurtosis": df[selected_var].kurtosis()
            }
            
            for stat, value in stats_summary.items():
                st.metric(stat, f"{value:.3f}")

# Clustering Analysis
elif analysis_type == "Clustering Analysis":
    st.markdown('<h2 class="sub-header">🔬 Clustering Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare data for clustering
    st.markdown("### District-wise Clustering")
    
    # Aggregate data by district
    district_data = df.groupby('district_name').agg({
        'amount_sanct': 'sum',
        'project_cost': 'sum',
        'subsidy_ratio': 'mean',
        'id': 'count'  # Number of projects
    }).rename(columns={'id': 'project_count'})
    
    # Remove rows with NaN values
    district_data = district_data.dropna()
    
    if len(district_data) > 0:
        # Feature selection for clustering
        features_for_clustering = st.multiselect(
            "Select Features for Clustering",
            district_data.columns.tolist(),
            default=['amount_sanct', 'project_count']
        )
        
        if len(features_for_clustering) >= 2:
            # Number of clusters
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            
            # Prepare data
            X = district_data[features_for_clustering].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to dataframe
            district_data['Cluster'] = clusters
            
            # Visualization
            if len(features_for_clustering) >= 2:
                fig = px.scatter(district_data, 
                               x=features_for_clustering[0], 
                               y=features_for_clustering[1],
                               color='Cluster',
                               hover_data=['project_count'] if 'project_count' in district_data.columns else None,
                               title=f"District Clustering: {features_for_clustering[0]} vs {features_for_clustering[1]}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster summary
            st.markdown("### Cluster Summary")
            cluster_summary = district_data.groupby('Cluster')[features_for_clustering].mean()
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Elbow method
            st.markdown("### Elbow Method for Optimal Clusters")
            inertias = []
            k_range = range(1, 11)
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42)
                kmeans_temp.fit(X_scaled)
                inertias.append(kmeans_temp.inertia_)
            
            fig = px.line(x=list(k_range), y=inertias, 
                         title="Elbow Method for Optimal Number of Clusters",
                         labels={'x': 'Number of Clusters', 'y': 'Inertia'})
            st.plotly_chart(fig, use_container_width=True)

# ML Models
elif analysis_type == "ML Models":
    st.markdown('<h2 class="sub-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Simple Linear Regression", "Multiple Linear Regression"])
    
    with tab1:
        st.markdown("### Simple Linear Regression")
        
        # Feature selection
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select Independent Variable (X)", numeric_columns, key="slr_x")
        with col2:
            y_feature = st.selectbox("Select Dependent Variable (Y)", numeric_columns, key="slr_y")
        
        if x_feature and y_feature and x_feature != y_feature:
            # Prepare data
            regression_data = df[[x_feature, y_feature]].dropna()
            
            if len(regression_data) > 10:
                X = regression_data[[x_feature]]
                y = regression_data[y_feature]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train R²", f"{train_r2:.3f}")
                with col2:
                    st.metric("Test R²", f"{test_r2:.3f}")
                with col3:
                    st.metric("Train RMSE", f"{train_rmse:.3f}")
                with col4:
                    st.metric("Test RMSE", f"{test_rmse:.3f}")
                
                # Scatter plot with regression line
                fig = px.scatter(regression_data, x=x_feature, y=y_feature,
                               title=f"Simple Linear Regression: {y_feature} vs {x_feature}")
                
                # Add regression line
                x_range = np.linspace(regression_data[x_feature].min(), 
                                    regression_data[x_feature].max(), 100)
                y_range = model.predict(x_range.reshape(-1, 1))
                
                fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', 
                                       name='Regression Line', line=dict(color='red')))
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual plot
                residuals = y_test - y_pred_test
                fig_residuals = px.scatter(x=y_pred_test, y=residuals,
                                         title="Residual Plot",
                                         labels={'x': 'Predicted Values', 'y': 'Residuals'})
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)
    
    with tab2:
        st.markdown("### Multiple Linear Regression")
        
        # Feature selection
        available_features = [col for col in numeric_columns if col != 'id']
        
        y_feature_mlr = st.selectbox("Select Target Variable (Y)", available_features, key="mlr_y")
        x_features_mlr = st.multiselect("Select Independent Variables (X)", 
                                       [col for col in available_features if col != y_feature_mlr],
                                       key="mlr_x")
        
        if y_feature_mlr and len(x_features_mlr) >= 2:
            # Prepare data
            features_list = x_features_mlr + [y_feature_mlr]
            regression_data_mlr = df[features_list].dropna()
            
            if len(regression_data_mlr) > 20:
                X_mlr = regression_data_mlr[x_features_mlr]
                y_mlr = regression_data_mlr[y_feature_mlr]
                
                # Split data
                X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
                    X_mlr, y_mlr, test_size=0.2, random_state=42)
                
                # Train model
                model_mlr = LinearRegression()
                model_mlr.fit(X_train_mlr, y_train_mlr)
                
                # Predictions
                y_pred_train_mlr = model_mlr.predict(X_train_mlr)
                y_pred_test_mlr = model_mlr.predict(X_test_mlr)
                
                # Metrics
                train_r2_mlr = r2_score(y_train_mlr, y_pred_train_mlr)
                test_r2_mlr = r2_score(y_test_mlr, y_pred_test_mlr)
                train_rmse_mlr = np.sqrt(mean_squared_error(y_train_mlr, y_pred_train_mlr))
                test_rmse_mlr = np.sqrt(mean_squared_error(y_test_mlr, y_pred_test_mlr))
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train R²", f"{train_r2_mlr:.3f}")
                with col2:
                    st.metric("Test R²", f"{test_r2_mlr:.3f}")
                with col3:
                    st.metric("Train RMSE", f"{train_rmse_mlr:.3f}")
                with col4:
                    st.metric("Test RMSE", f"{test_rmse_mlr:.3f}")
                
                # Feature importance
                st.markdown("#### Feature Importance (Coefficients)")
                importance_df = pd.DataFrame({
                    'Feature': x_features_mlr,
                    'Coefficient': model_mlr.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                fig_importance = px.bar(importance_df, x='Coefficient', y='Feature',
                                      orientation='h', title="Feature Importance")
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Actual vs Predicted
                fig_pred = px.scatter(x=y_test_mlr, y=y_pred_test_mlr,
                                    title="Actual vs Predicted Values",
                                    labels={'x': 'Actual', 'y': 'Predicted'})
                
                # Add perfect prediction line
                min_val = min(y_test_mlr.min(), y_pred_test_mlr.min())
                max_val = max(y_test_mlr.max(), y_pred_test_mlr.max())
                fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                            mode='lines', name='Perfect Prediction',
                                            line=dict(color='red', dash='dash')))
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.warning("Not enough data points for multiple linear regression analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    📊 Cold Chain Subsidy Dashboard | Built with Streamlit & Plotly
    </div>
    """, 
    unsafe_allow_html=True
)