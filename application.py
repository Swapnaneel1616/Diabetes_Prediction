import pickle 
from flask import Flask , request , jsonify , render_template
import numpy as np
import pandas as pd 
# The following imports are not strictly necessary for the web app but are good to keep for reference
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

# 1. Load the pickle file 
# NOTE: Ensure you have a directory named 'models' and the file 'regressor.pkl' inside it.
try:
    regressor_model = pickle.load(open('models/regressor.pkl' , 'rb'))
except FileNotFoundError:
    print("WARNING: Model file 'models/regressor.pkl' not found.")
    regressor_model = None # Use None if the model load fails

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('index.html')

# --- Prediction Logic ---
@app.route('/predictdata' , methods = ['GET' , 'POST'])
def predict_data():
    if request.method == "POST":
        
        try:
            # FIX: Remove the trailing commas (,) on each line.
            # They were turning each variable into a 1-element tuple, which causes the prediction to fail.
            Age = float(request.form.get('age'))
            Sex = float(request.form.get('sex'))
            BMI = float(request.form.get('bmi'))
            BP = float(request.form.get('bp'))
            S1 = float(request.form.get('s1'))
            S2 = float(request.form.get('s2'))
            S3 = float(request.form.get('s3'))
            S4 = float(request.form.get('s4'))
            S5 = float(request.form.get('s5'))
            S6 = float(request.form.get('s6'))

            # Combine all features into a single list
            all_features = [Age, Sex, BMI, BP, S1, S2, S3, S4, S5, S6]
            
            # Use np.array and reshape to ensure the model gets a 2D array: [[f1, f2, ...]]
            final_input = np.array(all_features).reshape(1, -1)

            # Make the prediction
            if regressor_model:
                prediction = regressor_model.predict(final_input)[0]
                # Format the prediction to two decimal places for clean display
                formatted_prediction = f"{prediction:.2f}"
            else:
                formatted_prediction = "Error: Model not loaded."
                
            # Render the home template and pass the result
            return render_template('home.html', final_result=formatted_prediction)
            
        except ValueError as e:
            # This handles cases where float() conversion fails (e.g., empty or non-numeric input)
            error_msg = "Input Error: Please ensure all 10 fields have valid numerical data."
            # Render the page, showing the error instead of the result
            return render_template('home.html', error=error_msg) 

        except Exception as e:
            # Generic error handler
            error_msg = f"An unexpected server error occurred during prediction: {e}"
            return render_template('home.html', error=error_msg)
            
    else:
        # GET request renders the form
        return render_template('home.html')

if __name__ == "__main__":
    # Use debug=True to display full error traceback in the browser for easy debugging
    app.run(host="0.0.0.0", debug=True)