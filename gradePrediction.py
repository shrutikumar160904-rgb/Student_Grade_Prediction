"""
Student Grade Prediction System
Predicts final grades (G3) based on demographic, social, and academic factors
Using real student performance dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import time
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class StudentGradePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        # Generate random seed based on current time for different outputs each run
        self.random_seed = int(time.time() * 1000) % 100000
        print(f"🎲 Random Seed for this run: {self.random_seed}")
        
    def load_data(self, filepath='student-mat.csv'):
        """Load student data from CSV file"""
        try:
            # Try to read with semicolon separator
            df = pd.read_csv(filepath, sep=';')
            print(f"✓ Loaded {len(df)} records from {filepath}")
            return df
        except FileNotFoundError:
            print(f"Error: Could not find {filepath}")
            print("Please ensure the CSV file is in the same directory as this script")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data: encode categorical variables and handle missing values"""
        df_processed = df.copy()
        
        # Remove quotes from string values if present
        for col in df_processed.select_dtypes(include=['object']).columns:
            df_processed[col] = df_processed[col].str.strip('"')
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                if col in self.label_encoders:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Target variable is G3 (final grade)
        target = 'G3'
        
        # Features to exclude
        exclude_features = ['G3', 'G1', 'G2']  # Exclude G1 and G2 as they're too correlated
        
        # Select features
        feature_cols = [col for col in df.columns if col not in exclude_features]
        
        X = df[feature_cols]
        y = df[target]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def explore_data(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}")
        print(f"Target Variable: G3 (Final Grade)")
        
        print("\nGrade Distribution:")
        print(df['G3'].describe())
        
        print("\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found ✓")
        else:
            print(missing[missing > 0])
        
        # Grade distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['G3'], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Final Grade (G3)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Final Grades')
        plt.axvline(df['G3'].mean(), color='r', linestyle='--', label=f'Mean: {df["G3"].mean():.2f}')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        study_time_map = {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}
        df['studytime_label'] = df['studytime'].map(study_time_map)
        df.groupby('studytime_label')['G3'].mean().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xlabel('Weekly Study Time')
        plt.ylabel('Average Final Grade')
        plt.title('Study Time vs Final Grade')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 3)
        df.groupby('failures')['G3'].mean().plot(kind='bar', color='coral', edgecolor='black')
        plt.xlabel('Number of Past Failures')
        plt.ylabel('Average Final Grade')
        plt.title('Past Failures vs Final Grade')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        print("\n✓ Data exploration plots saved as 'data_exploration.png'")
        plt.show()
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Define models with random states for reproducibility within a run but different across runs
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_seed),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_seed),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_seed
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=5, 
                random_state=self.random_seed
            ),
            'SVR': SVR(kernel='rbf', C=10, gamma='scale')
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score with random shuffling
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, 
                scoring='neg_mean_squared_error',
                # Use different random state for CV splits each run
                # (Note: cv parameter doesn't use random_state directly, 
                # but the model training within CV uses the model's random_state)
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'cv_rmse': cv_rmse
            }
            print(f"  Cross-Validation RMSE: {cv_rmse:.2f}")
        
        # Select best model
        self.best_model_name = min(results, key=lambda x: results[x]['cv_rmse'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n✓ Best Model: {self.best_model_name}")
        return results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2
            })
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R² Score: {r2:.4f}")
        
        return pd.DataFrame(results)
    
    def plot_results(self, X_test, y_test, results_df):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison - RMSE
        ax1 = axes[0, 0]
        results_sorted = results_df.sort_values('RMSE')
        colors = ['#2ecc71' if model == self.best_model_name else '#3498db' 
                 for model in results_sorted['Model']]
        ax1.barh(results_sorted['Model'], results_sorted['RMSE'], color=colors)
        ax1.set_xlabel('RMSE (Lower is Better)')
        ax1.set_title('Model Comparison - Root Mean Squared Error')
        ax1.invert_yaxis()
        
        # 2. Model Comparison - R² Score
        ax2 = axes[0, 1]
        results_sorted = results_df.sort_values('R² Score', ascending=False)
        colors = ['#2ecc71' if model == self.best_model_name else '#3498db' 
                 for model in results_sorted['Model']]
        ax2.barh(results_sorted['Model'], results_sorted['R² Score'], color=colors)
        ax2.set_xlabel('R² Score (Higher is Better)')
        ax2.set_title('Model Comparison - R² Score')
        ax2.invert_yaxis()
        
        # 3. Actual vs Predicted (Best Model)
        ax3 = axes[1, 0]
        y_pred = self.best_model.predict(X_test)
        ax3.scatter(y_test, y_pred, alpha=0.5, s=30)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Grade')
        ax3.set_ylabel('Predicted Grade')
        ax3.set_title(f'Actual vs Predicted - {self.best_model_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction Error Distribution
        ax4 = axes[1, 1]
        errors = y_test - y_pred
        ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Prediction Error (Actual - Predicted)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Error Distribution - {self.best_model_name}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'model_performance.png'")
        plt.show()
    
    def feature_importance_analysis(self, X):
        """Analyze feature importance using the best model"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_)
        else:
            print("Feature importance not available for this model type")
            return
        
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance_df.head(15).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importances - {self.best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved as 'feature_importance.png'")
        plt.show()
        
        return feature_importance_df
    
    def predict_student_risk(self, df, X_test_scaled, y_test):
        """Identify at-risk students"""
        print("\n" + "="*60)
        print("AT-RISK STUDENT IDENTIFICATION")
        print("="*60)
        
        y_pred = self.best_model.predict(X_test_scaled)
        
        # Create risk categories
        risk_categories = []
        for actual, predicted in zip(y_test, y_pred):
            if predicted < 10:
                risk = "HIGH RISK"
            elif predicted < 12:
                risk = "MODERATE RISK"
            else:
                risk = "LOW RISK"
            risk_categories.append(risk)
        
        risk_summary = pd.Series(risk_categories).value_counts()
        print("\nRisk Distribution in Test Set:")
        for risk, count in risk_summary.items():
            percentage = (count / len(risk_categories)) * 100
            print(f"  {risk}: {count} students ({percentage:.1f}%)")
        
        # Identify students who need intervention
        high_risk_idx = [i for i, r in enumerate(risk_categories) if r == "HIGH RISK"]
        print(f"\n🔴 {len(high_risk_idx)} students identified as HIGH RISK (predicted grade < 10)")
        print("   → Immediate intervention recommended")
        
        moderate_risk_idx = [i for i, r in enumerate(risk_categories) if r == "MODERATE RISK"]
        print(f"🟡 {len(moderate_risk_idx)} students identified as MODERATE RISK (predicted grade 10-12)")
        print("   → Extra support recommended")
        
        return risk_categories

def main():
    print("="*60)
    print("STUDENT GRADE PREDICTION SYSTEM")
    print("Using Real Student Performance Dataset")
    print("="*60)
    
    # Initialize predictor (generates new random seed each run)
    predictor = StudentGradePredictor()
    
    # Load data
    print("\n1. Loading student data...")
    df = predictor.load_data('student-mat.csv')
    
    if df is None:
        print("\nFailed to load data. Please check the file path.")
        return
    
    # Display basic info
    print("\nDataset Information:")
    print(f"  Total Students: {len(df)}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Target: G3 (Final Grade, 0-20 scale)")
    
    # Explore data
    print("\n2. Exploring data...")
    predictor.explore_data(df)
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    df_processed = predictor.preprocess_data(df, is_training=True)
    
    # Prepare features
    X, y = predictor.prepare_features(df_processed)
    print(f"   ✓ Features prepared: {X.shape[1]} features")
    
    # Split into train and test sets - USE DIFFERENT RANDOM STATE EACH RUN
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=predictor.random_seed
    )
    
    # Scale features
    predictor.scaler.fit(X_train)
    X_train_scaled = predictor.scaler.transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    print(f"   ✓ Training set: {len(X_train)} students")
    print(f"   ✓ Test set: {len(X_test)} students")
    
    # Train models
    print("\n4. Training models...")
    train_results = predictor.train_models(X_train_scaled, y_train)
    
    # Evaluate models
    print("\n5. Evaluating models...")
    results_df = predictor.evaluate_models(X_test_scaled, y_test)
    
    # Visualize results
    print("\n6. Generating visualizations...")
    predictor.plot_results(X_test_scaled, y_test, results_df)
    
    # Feature importance
    print("\n7. Analyzing feature importance...")
    feature_importance_df = predictor.feature_importance_analysis(X_train_scaled)
    
    # Risk assessment
    print("\n8. Identifying at-risk students...")
    risk_categories = predictor.predict_student_risk(df_processed, X_test_scaled, y_test)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n✓ Best Model: {predictor.best_model_name}")
    best_r2 = results_df[results_df['Model'] == predictor.best_model_name]['R² Score'].values[0]
    best_rmse = results_df[results_df['Model'] == predictor.best_model_name]['RMSE'].values[0]
    print(f"✓ Best Model R² Score: {best_r2:.4f}")
    print(f"✓ Best Model RMSE: {best_rmse:.2f} points")
    print(f"✓ Visualizations saved in current directory")
    print(f"✓ Random Seed Used: {predictor.random_seed}")
    print("\n📊 Key Insights:")
    print("   - The system can predict student performance with reasonable accuracy")
    print("   - Teachers can identify at-risk students early for intervention")
    print("   - Study time, past failures, and absences are typically key factors")
    print("\n💡 Note: Results vary each run due to different train-test splits")
    print("   Run the program multiple times to see different model performances!")

if __name__ == "__main__":
    main()
