import pickle 
from flask import Flask , request , jsonify , render_template
import numpy as np
import pandas as pd 
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

try:
    regressor_model = pickle.load(open('models/regressor.pkl' , 'rb'))
except FileNotFoundError:
    print("WARNING: Model file 'models/regressor.pkl' not found.")
    regressor_model = None 

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predictdata' , methods = ['GET' , 'POST'])
def predict_data():
    if request.method == "POST":
        
        try:
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


            all_features = [Age, Sex, BMI, BP, S1, S2, S3, S4, S5, S6]

            final_input = np.array(all_features).reshape(1, -1)

  
            if regressor_model:
                prediction = regressor_model.predict(final_input)[0]

                formatted_prediction = f"{prediction:.2f}"
            else:
                formatted_prediction = "Error: Model not loaded."
                

            return render_template('home.html', final_result=formatted_prediction)
            
        except ValueError as e:

            error_msg = "Input Error: Please ensure all 10 fields have valid numerical data."

            return render_template('home.html', error=error_msg) 

        except Exception as e:

            error_msg = f"An unexpected server error occurred during prediction: {e}"
            return render_template('home.html', error=error_msg)
            
    else:
        # GET request renders the form
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
