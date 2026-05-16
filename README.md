# Comparative Analysis of Hyperparameter Tuning

A clean and minimal Streamlit web application comparing `GridSearchCV` and `RandomizedSearchCV` on a Heart Disease dataset, focusing strictly on a Random Forest model.

## Features
- **Exploratory Data Analysis:** View dataset metrics, distributions, and correlation heatmaps.
- **Model Training:** Train a Random Forest classifier directly in the app.
- **Hyperparameter Tuning:** Choose between Grid Search and Random Search.
- **Performance Evaluation:** Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
- **Comparison Dashboards:** Compare the speed and accuracy of Grid Search vs Random Search vs Baseline side-by-side.

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## How to Run Locally

1. **Clone the repository** (or download the files).
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On Mac/Linux use: source venv/bin/activate
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure `heart.csv` is in the directory**. (If it is missing, the app will automatically fetch a fallback dataset from a public URL).
5. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

#
