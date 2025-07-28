import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =========================
# Data Loading & Inspection
# =========================
def load_and_preprocess_data(file_path):
    """
    Load and inspect the UNSW-NB15 dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Print basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Check data types
    print(f"\nData types:\n{df.dtypes}")
    
    return df

# =========================
# Feature Preparation
# =========================
def prepare_features(df):
    """
    Prepare features for anomaly detection (encode categoricals, select features)
    """
    print("\nPreparing features...")
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Encode categorical variables using LabelEncoder
    df_processed = df.copy()
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Select features (exclude known target columns)
    feature_cols = [col for col in df_processed.columns 
                   if col not in ['attack_cat', 'label', 'id']]
    X = df_processed[feature_cols]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features used: {feature_cols}")
    
    return X, feature_cols

# =========================
# Model Training
# =========================
def train_isolation_forest(X, contamination=0.1):
    """
    Train IsolationForest model for anomaly detection
    """
    print(f"\nTraining IsolationForest with contamination={contamination}...")
    
    # Initialize and fit IsolationForest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    iso_forest.fit(X)
    
    # Predict anomalies (-1: anomaly, 1: normal)
    predictions = iso_forest.predict(X)
    anomaly_scores = iso_forest.decision_function(X)
    # Convert to 1 for anomaly, 0 for normal
    anomaly_labels = (predictions == -1).astype(int)
    
    return iso_forest, anomaly_labels, anomaly_scores

# =========================
# Results Analysis
# =========================
def analyze_results(df, anomaly_labels, anomaly_scores):
    """
    Add anomaly results to DataFrame and print summary statistics
    """
    print("\nAnalyzing results...")
    
    # Add results to DataFrame
    df_results = df.copy()
    df_results['anomaly'] = anomaly_labels
    df_results['anomaly_score'] = anomaly_scores
    
    # Print stats
    n_anomalies = anomaly_labels.sum()
    n_normal = len(anomaly_labels) - n_anomalies
    anomaly_rate = n_anomalies / len(anomaly_labels)
    print(f"Total samples: {len(anomaly_labels)}")
    print(f"Anomalies detected: {n_anomalies}")
    print(f"Normal samples: {n_normal}")
    print(f"Anomaly rate: {anomaly_rate:.4f}")
    
    return df_results

# =========================
# Visualization
# =========================
def visualize_results(df_results, show=True):
    """
    Create and save visualizations for anomaly detection results
    """
    print("\nCreating visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram of anomaly scores
    axes[0, 0].hist(df_results['anomaly_score'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution of Anomaly Scores')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df_results['anomaly_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_results["anomaly_score"].mean():.3f}')
    axes[0, 0].legend()
    
    # Pie chart of anomaly vs normal
    anomaly_counts = df_results['anomaly'].value_counts()
    axes[0, 1].pie(anomaly_counts.values, labels=['Normal', 'Anomaly'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[0, 1].set_title('Anomaly Detection Results')
    
    # Boxplot of anomaly scores by class
    normal_scores = df_results[df_results['anomaly'] == 0]['anomaly_score']
    anomaly_scores = df_results[df_results['anomaly'] == 1]['anomaly_score']
    axes[1, 0].boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
    axes[1, 0].set_title('Anomaly Scores by Class')
    axes[1, 0].set_ylabel('Anomaly Score')
    
    # Cumulative distribution of anomaly scores
    sorted_scores = np.sort(df_results['anomaly_score'])
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 1].plot(sorted_scores, cumulative, linewidth=2)
    axes[1, 1].set_title('Cumulative Distribution of Anomaly Scores')
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

# =========================
# Feature Importance Analysis
# =========================
def feature_importance_analysis(df_results, feature_cols, show=True):
    """
    Analyze and plot feature importance for anomaly detection
    """
    print("\nAnalyzing feature importance...")
    # Only use numerical features for correlation
    numerical_features = df_results.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [col for col in numerical_features if col not in ['anomaly', 'anomaly_score']]
    
    feature_importance = {}
    for col in numerical_features:
        if col in df_results.columns:
            correlation = df_results[col].corr(df_results['anomaly_score'])
            feature_importance[col] = abs(correlation)
    
    # Sort and print top features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 most important features for anomaly detection:")
    for i, (feature, importance) in enumerate(sorted_features[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Plot top features
    top_features = sorted_features[:10]
    feature_names = [f[0] for f in top_features]
    importance_values = [f[1] for f in top_features]
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(feature_names)), importance_values, color='steelblue')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Absolute Correlation with Anomaly Score')
    plt.title('Feature Importance for Anomaly Detection')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return feature_importance

# =========================
# Main Pipeline
# =========================
def main():
    """
    Main function to run the anomaly detection pipeline
    """
    print("=== UNSW-NB15 Anomaly Detection using IsolationForest ===\n")
    
    file_path = 'UNSW-NB15_4.csv'
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        # Prepare features
        X, feature_cols = prepare_features(df)
        # Train IsolationForest
        iso_forest, anomaly_labels, anomaly_scores = train_isolation_forest(X, contamination=0.1)
        # Analyze results
        df_results = analyze_results(df, anomaly_labels, anomaly_scores)
        # Feature importance analysis (no plot yet)
        feature_importance = feature_importance_analysis(df_results, feature_cols, show=False)
        # Save results
        df_results.to_csv('anomaly_detection_results.csv', index=False)
        print(f"\nResults saved to 'anomaly_detection_results.csv'")
        print(f"Visualizations will be saved as 'anomaly_detection_results.png' and 'feature_importance.png'")
        # Print summary
        print("\n=== Summary ===")
        print(f"Dataset: {file_path}")
        print(f"Total samples: {len(df_results)}")
        print(f"Anomalies detected: {df_results['anomaly'].sum()}")
        print(f"Anomaly rate: {df_results['anomaly'].mean():.4f}")
        print(f"Model: IsolationForest with contamination=0.1")
        # Show plots at the end
        visualize_results(df_results, show=True)
        feature_importance_analysis(df_results, feature_cols, show=True)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
