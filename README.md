Diabetests_prediction 💉📈
Diabetes Progression Prediction using Decision Tree Regression
This project implements a machine learning model to predict the quantitative measure of diabetes disease progression one year after baseline, based on physiological and blood serum measurements. The model uses the well-known scikit-learn Diabetes dataset and employs a Decision Tree Regressor optimized using Grid Search Cross-Validation.

The final model is saved as a pickle file (regressor.pkl) for easy deployment into a web application (like the Flask app discussed previously).


⚙️ Repository Structure
.
├── Diabetes_Prediction.ipynb       # The main Jupyter Notebook (Data loading, EDA, Model Training, Hyperparameter Tuning)
├── regressor.pkl                   # The final, optimized Decision Tree Regressor model (for Flask deployment)
├── README.md                       # This file
└── (your Flask app files)          # Files like app.py, templates/home.html (if applicable)

📊 Dataset & Features
The model is trained on the Diabetes dataset loaded directly from sklearn.datasets.load_diabetes().

Number of Instances: 442

Target Variable (y): A quantitative measure of disease progression one year after baseline.

Features (X): 10 baseline variables which have been mean-centered and scaled (normalized) for modeling.

Feature Name,    Description,                          Note
age,        Age in years,                            Scaled
sex,        Sex,                                     Scaled
bmi,        Body Mass Index,                         Scaled
bp,        Average Blood Pressure,                   Scaled
s1,        tc (Total Serum Cholesterol),             Scaled
s2,        ldl (Low-Density Lipoproteins),           Scaled
s3,        hdl (High-Density Lipoproteins),          Scaled
s4,        tch (Total Cholesterol / HDL),            Scaled
s5,        ltg (Log of Serum Triglycerides),         Scaled
s6,        glu (Blood Sugar Level),                  Scaled



🧠 Model Training and Tuning
The model used for prediction is the Decision Tree Regressor. To find the best possible parameters, GridSearchCV was applied.

Hyperparameter Grid (param)
The following parameter ranges were explored:



{
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5, 10, 15, 20, 25],
    'max_features': ['auto', 'sqrt', 'log2']
}



Final Model Configuration
The final selectedmodel was trained using these optimized parameters.
📈 Model Performance
The final optimized model was evaluated on the test set (X_test and y_test).


Metric,Value,Interpretation
R2 Score,0.1068,"This R-squared value is quite low, indicating that the model only explains about 10.68% of the variance in the disease progression score. This suggests the linear model assumption of the underlying Decision Tree Regressor is not a great fit for this data, or the features alone have limited predictive power for this target."
Mean Absolute Error (MAE),63.77,"On average, the model's predictions are off by approximately 63.77 units of the disease progression score."
Mean Squared Error (MSE),5651.64,This large error value (the average squared difference between predicted and actual values) confirms the model's predictions have significant deviations from the actual target values.



🚀 Deployment File
The final, best-performing model (from GridSearchCV) was saved for immediate use in a deployment environment (like the Flask web app you are building).

import pickle
pickle.dump(regressor, open('regressor.pkl', 'wb'))



🛠️ Setup and Installation
git clone https://github.com/Swapnaneel1616/Diabetes_Prediction.git
cd Diabetes_Prediction

pip install pandas numpy scikit-learn seaborn matplotlib
