from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
app = Flask(__name__)

# Function to authenticate users (replace this with your actual authentication logic)
def authenticate(username, password):
    return username == "admin" and password == "admin"

@app.route('/login', methods=['GET', 'POST'])
def login():
    login_error = None  # Initialize login_error
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if authenticate(username, password):
            # Redirect to the index page with the username as a parameter
            return redirect(url_for('index', username=username))
        else:
            login_error = "Invalid username or password."

    return render_template('login.html', login_error=login_error)

# Initialize global variable to store the dataset
current_df = pd.DataFrame()
@app.route('/', methods=['GET', 'POST'])
def index():
    global current_df  # Use the global variable
    
    df_head = None
    target_column = None
    accuracy = None
    classification_report_str = None
    error_message = None

    if request.method == 'POST':
        try:
            # Load and preprocess the dataset
            dataset_url = request.form['dataset-url']
            id_column_name = request.form['id-column-name']
            target_column_name = request.form['target-column-name']
            test_size = float(request.form['test-size'])  # Convert user input to a float

            print(f"Attempting to load data from URL: {dataset_url}")
            path = 'https://drive.google.com/uc?export=download&id=' + dataset_url.split('/')[-2]

            # Reset current_df to an empty DataFrame
            current_df = pd.DataFrame()

            current_df = pd.read_csv(path)
            current_df.drop([id_column_name], axis=1, inplace=True)

            # Display the first few rows of the DataFrame
            df_head = current_df.head().to_html(classes='table table-striped')

            # Assume the target column is named 'label'
            target_column = current_df[target_column_name].to_string(index=False)

            # Split the data into training and testing sets with customizable test_size
            random_state = check_random_state(42)
            X = current_df.drop(target_column_name, axis=1)
            y = current_df[target_column_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Create and train the Gaussian Naive Bayes model
            clf = GaussianNB()
            clf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = clf.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            classification_report_str = classification_report(y_test, y_pred)

        except Exception as e:
            error_message = f"Error: {str(e)} Data tidak dapat diklasifikasikan."
            # Reset current_df to an empty DataFrame
            current_df = pd.DataFrame()

    return render_template('index.html', df_head=df_head, target_column=target_column, accuracy=accuracy,
                           classification_report=classification_report_str, error_message=error_message)

if __name__ == '__main__':
    np.random.seed(42)
    app.run(debug=True)
