import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Hyperparameter Tuning Analysis", layout="wide")

with st.container():
    # 1. Title and Introduction
    st.title("Comparative Analysis of Grid Search CV and Random Search CV")
    st.subheader("for Hyperparameter Tuning using Heart Disease Dataset")
    

    st.info("""
    **About Hyperparameter Tuning:**  
    Hyperparameter tuning helps optimize machine learning models by selecting the best parameter combinations to improve accuracy. It prevents overfitting and ensures the model generalizes well on unseen data.
    """)

# Load Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('heart.csv')
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
        df = pd.read_csv(url)
    return df

df = load_data()

with st.container():
    # 2. Sidebar
    st.sidebar.header("Configuration")
    selected_tuning = st.sidebar.selectbox("Select Hyperparameter Tuning Method for interactive test:", ("GridSearchCV", "RandomizedSearchCV"))

with st.container():
    # 3. Dataset Section
    st.header("1. Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    col_shape, col_dist = st.columns(2)
    with col_shape:
        st.subheader("Dataset Shape")
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

    with col_dist:
        st.subheader("Target Class Distribution")
        class_dist = df['target'].value_counts().rename_axis('Target').reset_index(name='Count')
        st.dataframe(class_dist, use_container_width=True)

with st.container():
    # 4. Exploratory Data Analysis
    st.header("2. Exploratory Data Analysis (EDA)")
    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.subheader("Class Distribution Graph")
        fig1, ax1 = plt.subplots(figsize=(4.5, 3))
        _ = sns.countplot(x='target', data=df, palette='Set2', ax=ax1)
        _ = ax1.set_title('Class Distribution (0 = No Disease, 1 = Disease)')
        st.pyplot(fig1)

    with col_eda2:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        _ = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax2, annot_kws={"size": 8})
        _ = ax2.set_title('Feature Correlation', fontsize=14)
        _ = ax2.tick_params(axis='x', labelsize=10)
        _ = ax2.tick_params(axis='y', labelsize=10)
        st.pyplot(fig2)

# Preprocessing
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

with st.container():
    # 5. Interactive Model Training and Results
    st.header("3. Interactive Model Training")
    st.write(f"Test a single run of **{selected_tuning}** on the Random Forest model.")

    if st.button("Run Model"):
        model = RandomForestClassifier(random_state=42)
        
        with st.spinner(f"Running {selected_tuning} for Random Forest..."):
            start_time = time.time()
            
            if selected_tuning == "GridSearchCV":
                search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            else:
                search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
                
            _ = search.fit(X_train_scaled, y_train)
            exec_time = time.time() - start_time
            
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.success("Model Training Complete!")
            
            st.subheader("Evaluation Metrics")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Accuracy", f"{acc:.3f}")
            col_m2.metric("Precision", f"{prec:.3f}")
            col_m3.metric("Recall", f"{rec:.3f}")
            col_m4.metric("F1-score", f"{f1:.3f}")
            col_m5.metric("Execution Time", f"{exec_time:.3f} s")
            
            st.subheader("Best Parameters Found")
            st.code(search.best_params_)
            
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(2.5, 2))
            cm = confusion_matrix(y_test, y_pred)
            _ = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm, annot_kws={"size": 9})
            _ = ax_cm.set_xlabel('Predicted', fontsize=9)
            _ = ax_cm.set_ylabel('Actual', fontsize=9)
            _ = ax_cm.set_title(f'Confusion Matrix ({selected_tuning})', fontsize=10)
            _ = ax_cm.tick_params(labelsize=8)
            st.pyplot(fig_cm)


with st.container():
    # 8. Comparison Section
    st.header("4. Baseline vs Grid Search vs Random Search")
    st.write("Click the button below to train Baseline Random Forest along with both tuning methods to compare performance.")

    @st.cache_data(show_spinner=False)
    def run_all_comparisons():
        results = []
        
        # 1. Baseline Model
        start = time.time()
        baseline = RandomForestClassifier(random_state=42)
        _ = baseline.fit(X_train_scaled, y_train)
        t_base = time.time() - start
        y_pred_base = baseline.predict(X_test_scaled)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        results.append({
            'Model Type': 'Baseline RF',
            'Test Accuracy': acc_base,
            'Precision': precision_score(y_test, y_pred_base),
            'Recall': recall_score(y_test, y_pred_base),
            'F1-score': f1_score(y_test, y_pred_base),
            'Execution Time (s)': t_base,
            'Best Params': "Default",
            'predictions': y_pred_base
        })
        
        # 2. GridSearch
        model_instance = RandomForestClassifier(random_state=42)
        start = time.time()
        gs = GridSearchCV(estimator=model_instance, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        _ = gs.fit(X_train_scaled, y_train)
        t_gs = time.time() - start
        y_pred_gs = gs.best_estimator_.predict(X_test_scaled)
        acc_gs = accuracy_score(y_test, y_pred_gs)
        
        results.append({
            'Model Type': 'GridSearch RF',
            'Test Accuracy': acc_gs,
            'Precision': precision_score(y_test, y_pred_gs),
            'Recall': recall_score(y_test, y_pred_gs),
            'F1-score': f1_score(y_test, y_pred_gs),
            'Execution Time (s)': t_gs,
            'Best Params': str(gs.best_params_),
            'predictions': y_pred_gs
        })
        
        # 3. RandomSearch
        start = time.time()
        rs = RandomizedSearchCV(estimator=model_instance, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        _ = rs.fit(X_train_scaled, y_train)
        t_rs = time.time() - start
        y_pred_rs = rs.best_estimator_.predict(X_test_scaled)
        acc_rs = accuracy_score(y_test, y_pred_rs)
        
        results.append({
            'Model Type': 'RandomSearch RF',
            'Test Accuracy': acc_rs,
            'Precision': precision_score(y_test, y_pred_rs),
            'Recall': recall_score(y_test, y_pred_rs),
            'F1-score': f1_score(y_test, y_pred_rs),
            'Execution Time (s)': t_rs,
            'Best Params': str(rs.best_params_),
            'predictions': y_pred_rs
        })
        
        return results

    if st.button("Generate Full Comparison"):
        with st.spinner("Running Baseline, GridSearchCV, and RandomizedSearchCV... This might take 5-10 seconds."):
            raw_results = run_all_comparisons()
            
        comp_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in raw_results])
        
        st.subheader("Comparison Table")
        st.dataframe(comp_df[['Model Type', 'Test Accuracy', 'Precision', 'Recall', 'F1-score', 'Execution Time (s)']])
        
        st.subheader("Best Parameters Found")
        st.write("**Best Parameters Found by GridSearchCV:**")
        st.code(comp_df[comp_df['Model Type'] == 'GridSearch RF']['Best Params'].values[0])
        
        st.write("**Best Parameters Found by RandomizedSearchCV:**")
        st.code(comp_df[comp_df['Model Type'] == 'RandomSearch RF']['Best Params'].values[0])
        
        st.subheader("Key Metrics Overview")
        col_c1, col_c2, col_c3 = st.columns(3)
        
        base_acc = comp_df[comp_df['Model Type'] == 'Baseline RF']['Test Accuracy'].values[0]
        grid_acc = comp_df[comp_df['Model Type'] == 'GridSearch RF']['Test Accuracy'].values[0]
        rand_acc = comp_df[comp_df['Model Type'] == 'RandomSearch RF']['Test Accuracy'].values[0]
        
        col_c1.metric("Baseline RF Accuracy", f"{base_acc:.3f}")
        col_c2.metric("GridSearch RF Accuracy", f"{grid_acc:.3f}", f"{(grid_acc - base_acc):.3f} vs Baseline")
        col_c3.metric("RandomSearch RF Accuracy", f"{rand_acc:.3f}", f"{(rand_acc - base_acc):.3f} vs Baseline")
        
        methods = comp_df['Model Type'].tolist()
        accuracies = comp_df['Test Accuracy'].tolist()
        times = comp_df['Execution Time (s)'].tolist()
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Accuracy Comparison Graph")
            fig_acc, ax_acc = plt.subplots(figsize=(4.5, 3))
            bars = ax_acc.bar(methods, accuracies, color=['lightgrey', 'skyblue', 'salmon'], width=0.5)
            _ = ax_acc.set_ylabel('Test Accuracy', fontsize=9)
            _ = ax_acc.set_ylim(0, 1.1)
            _ = ax_acc.tick_params(labelsize=8)
            for p in bars:
                _ = ax_acc.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width()/2., p.get_height() * 1.01), ha='center', fontsize=9)
            st.pyplot(fig_acc)
            
        with col_g2:
            st.subheader("Execution Time Comparison Graph")
            fig_time, ax_time = plt.subplots(figsize=(4.5, 3))
            bars2 = ax_time.bar(methods, times, color=['lightgrey', 'lightblue', 'lightcoral'], width=0.5)
            _ = ax_time.set_ylabel('Time (Seconds)', fontsize=9)
            _ = ax_time.tick_params(labelsize=8)
            for p in bars2:
                _ = ax_time.annotate(f'{p.get_height():.2f}s', (p.get_x() + p.get_width()/2., p.get_height() * 1.01), ha='center', fontsize=9)
            st.pyplot(fig_time)
            
        st.subheader("Confusion Matrix (Best Model)")
        st.write("Displaying the Confusion Matrix for the model with the highest test accuracy.")
        best_model_idx = np.argmax(accuracies)
        best_model_name = methods[best_model_idx]
        best_predictions = raw_results[best_model_idx]['predictions']
        
        fig_best_cm, ax_best_cm = plt.subplots(figsize=(2.5, 2))
        cm_best = confusion_matrix(y_test, best_predictions)
        _ = sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_best_cm, annot_kws={"size": 9})
        _ = ax_best_cm.set_xlabel('Predicted', fontsize=9)
        _ = ax_best_cm.set_ylabel('Actual', fontsize=9)
        _ = ax_best_cm.set_title(f'Confusion Matrix - {best_model_name}', fontsize=10)
        _ = ax_best_cm.tick_params(labelsize=8)
        st.pyplot(fig_best_cm)

with st.container():
    # 9. Final Conclusion
    st.header("5. Conclusion")
    st.markdown("""
    * **Tuning Improvement:** By comparing Baseline RF with the tuned models, we can clearly observe how tuning the hyperparameters impacts the accuracy and generalization of the model.
    * **Accuracy:** Both `GridSearchCV` and `RandomizedSearchCV` often yield similar or identical accuracy results on the Random Forest model.
    * **Execution Time:** **`RandomizedSearchCV` is significantly faster** than `GridSearchCV`. Since Random Forest has many combinations of hyperparameters (trees, depth, split criteria), testing them all exhaustively takes a lot of time.
    * **Recommendation:** For a complex algorithm like Random Forest, `RandomizedSearchCV` provides an excellent balance—yielding near-optimal performance in a fraction of the time.
    """)
