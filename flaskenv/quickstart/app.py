from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Inisialisasi variabel global untuk menyimpan dataset
current_df = pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_df  # Gunakan variabel global

    df_head = None
    target_column = None
    accuracy = None
    classification_report_str = None
    error_message = None

    if request.method == 'POST':
        try:
            # Load and preprocess the dataset
            dataset_url = request.form['dataset-url']
            path = 'https://drive.google.com/uc?export=download&id='+dataset_url.split('/')[-2]

            # Bersihkan data lama
            current_df = pd.DataFrame()

            current_df = pd.read_csv(path)
            current_df.drop(['Car_Name'], axis=1, inplace=True)

            # Display the first few rows of the DataFrame
            df_head = current_df.head().to_html(classes='table table-striped')

            # Assume the target column is named 'label'
            target_column = current_df['Owner'].to_string(index=False)

            # Split the data into training and testing sets
            X = current_df.drop('Owner', axis=1)
            y = current_df['Owner']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

    return render_template('index.html', df_head=df_head, target_column=target_column, accuracy=accuracy, classification_report=classification_report_str, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
