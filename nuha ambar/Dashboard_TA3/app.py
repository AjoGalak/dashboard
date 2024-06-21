import io
from io import BytesIO
import logging
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, url_for)
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__)

# variabel global untuk klasifikasi 1
training_data = None  
predictions_df = None  
model = None  
encoder = None
scaler = StandardScaler()  # Inisialisasi scaler
accuracy1 = None  
accuracy2 = None  

# variabel global untuk klasifikasi 2
model2 = None
preprocessor2 = None
training_data2 = None
predictions_df2 = None
accuracy2_train = None
accuracy2_test = None
y_test2 = None
y_test_pred2 = None
X2 = None
y2 = None


#train model klasifikasi 1
def train_model():
    global model, encoder, scaler, accuracy1, accuracy2

    # Initialize variables to None
    df = None
    train_accuracy = None
    test_accuracy = None

    try:
        df = pd.read_excel('file/data_fix.xlsx')
        X = df[['Gender', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6']]
        y = df['Lulus']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        encoder = OneHotEncoder(sparse_output=False, drop='first')
        gender_encoded = encoder.fit_transform(X_train[['Gender']])
        X_train_encoded = X_train.drop(['Gender'], axis=1)
        X_train_encoded = pd.concat([pd.DataFrame(gender_encoded, columns=['WANITA']), X_train_encoded.reset_index(drop=True)], axis=1)
        
        gender_encoded_test = encoder.transform(X_test[['Gender']])
        X_test_encoded = X_test.drop(['Gender'], axis=1)
        X_test_encoded = pd.concat([pd.DataFrame(gender_encoded_test, columns=['WANITA']), X_test_encoded.reset_index(drop=True)], axis=1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)

        model = GaussianNB()
        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        df['Prediksi'] = model.predict(scaler.transform(pd.concat([pd.DataFrame(encoder.transform(df[['Gender']]), columns=['WANITA']), df.drop(['Gender'], axis=1).reset_index(drop=True)], axis=1)))
        df['Prediksi'] = df['Prediksi'].map({1: 'On Time', 0: 'Late'})

        accuracy1 = train_accuracy
        accuracy2 = test_accuracy
    
    except KeyError as ke:
        logging.error(f"Key error: {ke}. Please check the columns in the uploaded file.")
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
    
    return model, encoder, scaler, df, test_accuracy, train_accuracy

model, encoder, scaler, df, test_accuracy, train_accuracy = train_model()

# Menampilkan hasil akurasi
if train_accuracy is not None and test_accuracy is not None:
    print(f"Akurasi pada data latih (train): {train_accuracy}")
    print(f"Akurasi pada data uji (test): {test_accuracy}")
else:
    print("Terdapat kesalahan dalam melatih model. Periksa log untuk detail lebih lanjut.")

#train model klasifikasi 2
def train_model_klasifikasi2():
    global model2, preprocessor2, accuracy2_train, accuracy2_test, training_data2, y_test2, y_test_pred2, X2, y2
    try:
        df = pd.read_excel("file/Z_dataset_all_angkatan_FIX (SID).xlsx")

        # Strip whitespace from headers and rename columns to remove extra spaces
        df.columns = df.columns.str.strip()
        # Split data into predictors (X) and target (y)
        X = df[['Gender', 'Lama Masa Studi', 'IPK', 'Score EPRT', 'Pengalaman Lomba', 'TAK', 'Lama Waktu Mendapatkan Pekerjaan']]
        y = df['Masa Tunggu']

        X2 = X
        y2 = y
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocessing and training the model
        preprocessor2 = ColumnTransformer([
            ('num', StandardScaler(), ['Lama Masa Studi', 'IPK', 'Score EPRT', 'TAK', 'Lama Waktu Mendapatkan Pekerjaan']),
            ('cat', OneHotEncoder(drop='first'), ['Gender', 'Pengalaman Lomba'])
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor2),
            ('classifier', GaussianNB())
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Save the model
        dump(pipeline, 'file/fix_prediksi_WT_modell.joblib')
        
        # Evaluate the model
        y_train_pred = pipeline.predict(X_train)
        accuracy2_train = accuracy_score(y_train, y_train_pred)
        
        y_test_pred = pipeline.predict(X_test)
        accuracy2_test = accuracy_score(y_test, y_test_pred)
        
        # Update global training data
        training_data2 = df.to_dict(orient='records')

        y_test2 = y_test
        y_test_pred2 = y_test_pred
        
        # Assign model2 to pipeline
        model2 = pipeline
        
        return pipeline, training_data2, accuracy2_train, accuracy2_test, y_test2, y_test_pred2, X2, y2

    except Exception as e:
        logging.error(f"Error in train_model_klasifikasi2: {e}")
        return None, None, None, None

train_model_klasifikasi2()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/klasifikasi1')
def klasifikasi1():
    return render_template('klasifikasi1.html')

@app.route('/klasifikasi2')
def klasifikasi2():
    return render_template('klasifikasi2.html')

@app.route('/manual_input')
def manual_input():
    return render_template('manual_input.html')

@app.route('/manual_input_klasifikasi2')
def manual_input_klasifikasi2():
    return render_template('klasifikasi_2/manual_input_klasifikasi2.html')

@app.route('/upload_training')
def upload_training():
    return render_template('upload_training.html')

@app.route('/upload_training_klasifikasi2')
def upload_training_klasifikasi2():
    return render_template('klasifikasi_2/upload_training_klasifikasi2.html')

# upload data training klasifikasi 1
@app.route('/upload_file_klasifikasi1', methods=['POST'])
def upload_file_klasifikasi1():
    global training_data, model, encoder, scaler, accuracy1, accuracy2
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."})
            
            model, encoder, scaler, training_data, accuracy1, accuracy2 = train_model(df)
            training_data = df.to_dict(orient='records')
            return redirect(url_for('show_training'))
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}")
            return jsonify({"error": "Failed to process file"})

# upload data training klasifikasi 2
@app.route('/upload_file_klasifikasi2', methods=['POST'])
def upload_file_klasifikasi2():
    global training_data2, model2, preprocessor2, accuracy2_train, accuracy2_test
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."})
            
            model2, training_data2, accuracy2_train, accuracy2_test = train_model_klasifikasi2(df)
            return redirect(url_for('show_training_klasifikasi2'))
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}")
            return jsonify({"error": "Failed to process file"})

# Menampilkan data training klasifikasi 1
@app.route('/get_training_data', methods=['GET'])
def get_training_data():
    global training_data, accuracy1
    if training_data is not None:
        return jsonify({"data": training_data, "accuracy": accuracy1})
    else:
        return jsonify({"error": "No training data available"}), 400

# Menampilkan data training klasifikasi 2
@app.route('/get_training_data2', methods=['GET'])
def get_training_data2():
    global training_data2, accuracy2_train
    if training_data2 is not None:
        return jsonify({"data": training_data2, "accuracy": accuracy2_train})
    else:
        return jsonify({"error": "No training data available"}), 400
    
@app.route('/show_training')
def show_training():
    global accuracy1, accuracy2
    return render_template('show_training.html', training_data=training_data, accuracy_train=accuracy1, accuracy_test=accuracy2)

@app.route('/show_training_klasifikasi2')
def show_training_klasifikasi2():
    global accuracy2_train, accuracy2_test
    return render_template('klasifikasi_2/show_training_klasifikasi2.html', training_data=training_data2, accuracy_train=accuracy2_train, accuracy_test=accuracy2_test)

@app.route('/show_accuracy_klasifikasi1', methods=['GET'])
def show_accuracy_klasifikasi1():
    if train_accuracy is not None and test_accuracy is not None:
        return jsonify({"accuracy_train": round(train_accuracy * 100, 2), "accuracy_test": round(test_accuracy * 100, 2)})
    else:
        return jsonify({"error": "Accuracy values not available"}), 400
    
@app.route('/show_accuracy_klasifikasi2', methods=['GET'])
def show_accuracy_klasifikasi2():
    global accuracy2_train, accuracy2_test
    if accuracy2_train is not None and accuracy2_test is not None:
        return jsonify({"accuracy_train": round(accuracy2_train * 100, 0), "accuracy_test": round(accuracy2_test * 100, 0)})
    else:
        return jsonify({"error": "Accuracy values not available"}), 400


# Upload data testing klasifikasi 1
@app.route('/upload_testing')
def upload_testing():
    return render_template('upload_testing.html')

@app.route('/upload_testing_klasifikasi1', methods=['POST'])
def upload_testing_klasifikasi1():
    global predictions_df, model, encoder, scaler, accuracy1, accuracy2
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."})
            
            logging.debug(f"Data from file:\n{df}")

            # Strip any leading or trailing spaces from column names
            df.columns = df.columns.str.strip()

            if 'Gender' not in df.columns:
                return jsonify({"error": "'Gender' column not found in uploaded file"}), 400

            # Ensure the 'SID' column is excluded from features
            features = ['Gender', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6']

            if not all(col in df.columns for col in features):
                missing_cols = [col for col in features if col not in df.columns]
                return jsonify({"error": f"Missing columns in uploaded file: {', '.join(missing_cols)}"}), 400

            # Encode the 'Gender' column
            gender_encoded = encoder.transform(df[['Gender']])
            X_encoded = df.drop(['Gender'], axis=1)
            X_encoded = X_encoded[features[1:]].reset_index(drop=True)  # Ensure only the necessary columns are used
            X_encoded = pd.concat([pd.DataFrame(gender_encoded, columns=['WANITA']), X_encoded], axis=1)

            logging.debug(f"Encoded data:\n{X_encoded}")

            # Scale the data
            X_scaled = scaler.transform(X_encoded)

            logging.debug(f"Scaled data:\n{X_scaled}")

            # Predict using the trained model
            predictions = model.predict(X_scaled)
            logging.debug(f"Raw predictions:\n{predictions}")

            prediction_mapping = {'TW': 'On Time', 'TTW': 'Late'}
            df['Prediction'] = predictions
            df['Prediction'] = df['Prediction'].map(prediction_mapping)

            logging.debug(f"Predictions with labels:\n{df['Prediction']}")
            logging.info(f"Training accuracy: {accuracy1}")
            logging.info(f"Testing accuracy: {accuracy2}")


            predictions_df = pd.DataFrame(df)

            return redirect(url_for('show_prediction'))
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}")
            return jsonify({"error": "Failed to process file"})

# Upload data testing klasifikasi 2
@app.route('/upload_testing_klasifikasi2')
def upload_testing_klasifikasi2():
    return render_template('klasifikasi_2/upload_testing_klasifikasi2.html')

@app.route('/upload_testing_file_klasifikasi2', methods=['POST'])
def upload_testing_file_klasifikasi2():
    global predictions_df2, model2, preprocessor2
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."})
            
            logging.debug(f"Data from file:\n{df}")

            # Strip any leading or trailing spaces from column names
            df.columns = df.columns.str.strip()

            # Print the uploaded data to verify
            logging.debug(f"Uploaded data:\n{df}")

            if 'Gender' not in df.columns:
                return jsonify({"error": "'Gender' column not found in uploaded file"}), 400

            # Define the features used in the model
            features = ['Gender', 'Lama Masa Studi', 'IPK', 'Score EPRT', 'Pengalaman Lomba', 'TAK', 'Lama Waktu Mendapatkan Pekerjaan']

            if not all(col in df.columns for col in features):
                missing_cols = [col for col in features if col not in df.columns]
                return jsonify({"error": f"Missing columns in uploaded file: {', '.join(missing_cols)}"}), 400

            # Ensure that preprocessor2 and model2 are initialized
            if preprocessor2 is None or model2 is None:
                return jsonify({"error": "Model not initialized. Please retrain the model."}), 500

            # Preprocess and predict
            X_preprocessed = preprocessor2.transform(df[features])
            predictions = model2.named_steps['classifier'].predict(X_preprocessed)
            
            df['Prediction'] = predictions

            # Assign predictions_df2
            predictions_df2 = df

            return redirect(url_for('show_prediction_klasifikasi2'))
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}")
            return jsonify({"error": "Failed to process file"}), 500

# Menampilkan hasil klasifikasi prediksi data testing klasifikasi 1
@app.route('/get_testing_data', methods=['GET'])
def get_testing_data():
    global predictions_df
    
    # Pastikan predictions_df telah diinisialisasi dengan data
    if predictions_df is not None:
        filter_value = request.args.get('filter', 'all')

        if filter_value == 'on_time':
            filtered_data = predictions_df[predictions_df['Prediction'] == 'On Time']
        elif filter_value == 'late':
            filtered_data = predictions_df[predictions_df['Prediction'] == 'Late']
        else:
            filtered_data = predictions_df

        # Debug print untuk memeriksa hasil filter
        print(filtered_data)

        if filtered_data.empty:
            return jsonify({"error": "No data available for the selected filter"}), 400
        
        predictions_list = filtered_data.to_dict(orient='records')
        return jsonify(predictions_list)
    else:
        return jsonify({"error": "No testing data available"}), 400


# Menampilkan hasil prediksi data testing klasifikasi 2
@app.route('/get_testing_klasifikasi2', methods=['GET'])
def get_testing_klasifikasi2():
    global predictions_df2
    if predictions_df2 is not None:
        filter_value = request.args.get('filter', 'all')

        if filter_value == 'cepat':
            filtered_data = predictions_df2[predictions_df2['Prediction'] == 'CEPAT']
        elif filter_value == 'lambat':
            filtered_data = predictions_df2[predictions_df2['Prediction'] == 'LAMBAT']
        else:
            filtered_data = predictions_df2

        # Debug print untuk memeriksa hasil filter
        print(f"Filter value: {filter_value}")
        print(f"Filtered data:\n{filtered_data}")

        if filtered_data.empty:
            return jsonify({"error": "No data available for the selected filter"}), 400
        
        predictions_list = filtered_data.to_dict(orient='records')
        return jsonify(predictions_list)
    else:
        return jsonify({"error": "No testing data available"}), 400


@app.route('/show_prediction')
def show_prediction():
    return render_template('show_prediction.html')

@app.route('/show_prediction_klasifikasi2')
def show_prediction_klasifikasi2():
    return render_template('klasifikasi_2/show_prediction_klasifikasi2.html')

@app.route('/view_plot')
def view_plot():
    return render_template('view_plot.html')

@app.route('/view_plot_klasifikasi2')
def view_plot_klasifikasi2():
    return render_template('klasifikasi_2/view_plot_klasifikasi2.html')

@app.route('/view_confusion_matrix')
def view_confusion_matrix():
    return render_template('klasifikasi_2/view_confusion_matrix.html')

@app.route('/view_heatmap')
def view_heatmap():
    return render_template('klasifikasi_2/view_heatmap.html')

# Training data dan hasil prediksi pada manual predict klasifikasi 1
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = pd.read_excel("file/data_fix.xlsx")

    # Preprocess data
    X = data[['Gender', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6']]
    y = data['Lulus'].apply(lambda x: 1 if x == 'TW' else 0)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode gender
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_train_encoded = encoder.fit_transform(X_train[['Gender']])
    X_test_encoded = encoder.transform(X_test[['Gender']])

    # Drop original gender column and add encoded gender columns
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=['PRIA'])
    X_train = X_train.drop('Gender', axis=1).reset_index(drop=True)
    X_train = pd.concat([X_train_encoded, X_train], axis=1)

    X_test_encoded = pd.DataFrame(X_test_encoded, columns=['PRIA'])
    X_test = X_test.drop('Gender', axis=1).reset_index(drop=True)
    X_test = pd.concat([X_test_encoded, X_test], axis=1)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    # Get form data
    form_data = request.get_json()
    gender = form_data['gender']
    ips1 = float(form_data['ips1'])
    ips2 = float(form_data['ips2'])
    ips3 = float(form_data['ips3'])
    ips4 = float(form_data['ips4'])
    ips5 = float(form_data['ips5'])
    ips6 = float(form_data['ips6'])

    # Convert gender to numeric
    gender_num = 1 if gender == 'PRIA' else 0

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[gender_num, ips1, ips2, ips3, ips4, ips5, ips6]],
                              columns=['PRIA', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6'])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_scaled)
    prediction_label = 'On Time' if prediction[0] == 1 else 'Late'

    # Hitung akurasi model
    y_pred_train = model.predict(X_train_scaled)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = model.predict(X_test_scaled)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    return jsonify({'prediction': prediction_label, 'accuracy_train': round(accuracy_train * 100, 2), 'accuracy_test': round(accuracy_test * 100, 2)})

# Training data dan hasil prediksi pada manual predict klasifikasi 2
@app.route('/predict_manual_klasifikasi2', methods=['POST'])
def predict_manual_klasifikasi2():
    # Load training data
    data = pd.read_excel("file/Z_dataset_all_angkatan_FIX.xlsx")

    # Preprocess data
    X = data[['Gender', 'Lama Masa Studi', 'IPK', 'Score EPRT', 'Pengalaman Lomba', 'TAK', 'Lama Waktu Mendapatkan Pekerjaan']]
    y = data['Masa Tunggu'].apply(lambda x: 1 if x == 'CEPAT' else 0)

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('gender', OneHotEncoder(drop='first'), ['Gender']),
            ('pengalaman_lomba', OneHotEncoder(drop='first'), ['Pengalaman Lomba']),
            ('numeric', StandardScaler(), ['Lama Masa Studi', 'IPK', 'Score EPRT', 'TAK', 'Lama Waktu Mendapatkan Pekerjaan'])
        ],
        remainder='passthrough'
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Get form data
    form_data = request.get_json()
    gender = form_data['gender']
    lama_masa_studi = int(form_data['lama_masa_studi'])
    ipk = float(form_data['ipk'])
    score_eprt = int(form_data['score_eprt'])
    pengalaman_lomba = form_data['pengalaman_lomba']
    tak = int(form_data['tak'])
    lama_waktu_mendapatkan_pekerjaan = float(form_data['lama_waktu_mendapatkan_pekerjaan'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[gender, lama_masa_studi, ipk, score_eprt, pengalaman_lomba, tak, lama_waktu_mendapatkan_pekerjaan]],
                              columns=['Gender', 'Lama Masa Studi', 'IPK', 'Score EPRT', 'Pengalaman Lomba', 'TAK', 'Lama Waktu Mendapatkan Pekerjaan'])

    # Normalize the input data
    X_input_scaled = pipeline.named_steps['preprocessor'].transform(input_data)

    # Make a prediction
    y_pred_input = pipeline.named_steps['classifier'].predict(X_input_scaled)
    prediction_label = 'CEPAT' if y_pred_input[0] == 1 else 'LAMBAT'

    # Calculate accuracy
    y_train_pred = pipeline.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    
    y_test_pred = pipeline.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    return jsonify({'prediction': prediction_label, 'accuracy_train': round(accuracy_train * 100, 0), 'accuracy_test': round(accuracy_test * 100, 0)})

@app.route('/download_predictions', methods=['GET'])
def download_predictions():
    global predictions_df
    if predictions_df is not None and not predictions_df.empty:
        try:
            # Convert the predictions to a DataFrame
            df = pd.DataFrame(predictions_df)
            
            # Debug DataFrame content
            logging.debug(f"DataFrame to be saved:\n{df}")
            
            # Define the file path
            output_file = 'predictions.xlsx'  # Simplify path for debugging purposes
            
            # Save the DataFrame to an Excel file
            df.to_excel(output_file, index=False)
            
            # Log file save success
            logging.debug(f"File saved to {output_file}")
            
            return send_file(output_file, as_attachment=True, download_name='predictions.xlsx')
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return jsonify({"error": "Failed to save and download file"}), 500
    else:
        return jsonify({"error": "No predictions available or predictions DataFrame is empty"}), 400

@app.route('/download_predictions_klasifikasi2', methods=['GET'])
def download_predictions_klasifikasi2():
    global predictions_df2
    if predictions_df2 is not None and not predictions_df2.empty:
        try:
            # Convert the predictions to a DataFrame
            df = pd.DataFrame(predictions_df2)
            
            # Debug DataFrame content
            logging.debug(f"DataFrame to be saved:\n{df}")
            
            # Define the file path
            output_file = 'predictions.xlsx'  # Simplify path for debugging purposes
            
            # Save the DataFrame to an Excel file
            df.to_excel(output_file, index=False)
            
            # Log file save success
            logging.debug(f"File saved to {output_file}")
            
            return send_file(output_file, as_attachment=True, download_name='predictions2.xlsx')
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return jsonify({"error": "Failed to save and download file"}), 500
    else:
        return jsonify({"error": "No predictions available or predictions DataFrame is empty"}), 400
    
@app.route('/plot_manual_prediction', methods=['POST'])
def plot_manual_prediction():
    # Mendapatkan data IPS yang diprediksi dari request
    form_data = request.get_json()
    ips_values = [float(form_data['ips1']), float(form_data['ips2']), float(form_data['ips3']), float(form_data['ips4']), float(form_data['ips5']), float(form_data['ips6'])]

    # Buat plot dari data IPS yang diprediksi
    fig, ax = plt.subplots()
    ax.plot(['IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6'], ips_values, marker='o', linestyle='-')
    ax.set_xlabel('Subjects')
    ax.set_ylabel('IPS Values')
    ax.set_title('IPS Values Prediction')

    # Simpan plot ke dalam BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    # Mengirimkan plot sebagai response
    return send_file(img, mimetype='image/png')

@app.route('/plot_manual_prediction_klasifikasi2', methods=['POST'])
def plot_manual_prediction_klasifikasi2():
    # Mendapatkan data input dari request
    form_data = request.get_json()
    lama_masa_studi = float(form_data['lama_masa_studi'])
    ipk = float(form_data['ipk'])
    score_eprt = float(form_data['score_eprt'])
    pengalaman_lomba = form_data['pengalaman_lomba']
    tak = float(form_data['tak'])
    lama_waktu_mendapatkan_pekerjaan = float(form_data['lama_waktu_mendapatkan_pekerjaan'])

    # Mengubah string 'Ada' menjadi 1 dan 'Tidak Ada' menjadi 0
    pengalaman_lomba_value = 1 if pengalaman_lomba == 'Ada' else 0

    # Buat subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot bar chart untuk Pengalaman Lomba
    ax1.bar(['Pengalaman Lomba'], [pengalaman_lomba_value], color='blue')
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Value')
    ax1.set_title('Pengalaman Lomba')

    # Plot scatter plot untuk Lama Masa Studi
    features = ['IPK', 'Score EPRT', 'TAK', 'Lama Waktu Mendapatkan Pekerjaan']
    values = [ipk, score_eprt, tak, lama_waktu_mendapatkan_pekerjaan]
    ax2.scatter([lama_masa_studi]*len(features), values)
    for i, feature in enumerate(features):
        ax2.text(lama_masa_studi, values[i], feature, fontsize=9, ha='right')

    ax2.set_xlabel('Lama Masa Studi')
    ax2.set_ylabel('Values')
    ax2.set_title('Lama Masa Studi vs Other Features')

    # Rotasi label sumbu x dan tambahkan margin bawah untuk kedua plot
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.25)

    # Simpan plot ke dalam BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    # Mengirimkan plot sebagai response
    return send_file(img, mimetype='image/png')

@app.route('/plot_predictions')
def plot_predictions():
    global predictions_df
    if predictions_df is not None:
        if isinstance(predictions_df, list):
            predictions_df = pd.DataFrame(predictions_df)

        filter_value = request.args.get('filter', 'all')

        if filter_value == 'on_time':
            filtered_data = predictions_df[predictions_df['Prediction'] == 'On Time']
        elif filter_value == 'late':
            filtered_data = predictions_df[predictions_df['Prediction'] == 'Late']
        else:
            filtered_data = predictions_df

        if 'Prediction' in predictions_df.columns and 'Gender' in predictions_df.columns:
            fig, ax = plt.subplots()

            # Group by Prediction and Gender
            grouped_data = filtered_data.groupby(['Prediction', 'Gender']).size().unstack(fill_value=0)

            # Plot the grouped data
            grouped_data.plot(kind='bar', ax=ax, stacked=False, color=['#28a745', '#1f77b4'])
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Results by Gender')

            # Set x-axis labels to be horizontal
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            # Add legend
            ax.legend(title='Gender')

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close(fig)
            return send_file(img, mimetype='image/png')
        else:
            return jsonify({"error": "'Prediction' or 'Gender' column not found in data"}), 400
    else:
        return jsonify({"error": "No predictions available to plot"}), 400

    
@app.route('/plot_predictions_klasifikasi2')
def plot_predictions_klasifikasi2():
    global predictions_df2
    if predictions_df2 is not None:
        if isinstance(predictions_df2, list):
            predictions_df2 = pd.DataFrame(predictions_df2)
        
        if 'Prediction' in predictions_df2.columns:
            filter_value = request.args.get('filter', 'all')

            # Apply filter to predictions_df2
            if filter_value == 'cepat':
                filtered_data = predictions_df2[predictions_df2['Prediction'] == 'CEPAT']
            elif filter_value == 'lambat':
                filtered_data = predictions_df2[predictions_df2['Prediction'] == 'LAMBAT']
            else:
                filtered_data = predictions_df2

            plot_count = 0
            fig, axs = plt.subplots(1, 2, figsize=(8, 6))
            
            # Plot bar chart gender
            if 'Gender' in filtered_data.columns and not filtered_data.empty:
                grouped_data_gender = filtered_data.groupby(['Prediction', 'Gender']).size().unstack(fill_value=0)
                grouped_data_gender.plot(kind='bar', ax=axs[plot_count], stacked=False, color=['#28a745', '#1f77b4'])
                axs[plot_count].set_xlabel('Prediction')
                axs[plot_count].set_ylabel('Count')
                axs[plot_count].set_title('Prediction Results by Gender')
                axs[plot_count].set_xticklabels(axs[plot_count].get_xticklabels(), rotation=0)
                axs[plot_count].legend(title='Gender')
                plot_count += 1

            # Plot bar chart pengalaman lomba
            if 'Pengalaman Lomba' in filtered_data.columns and not filtered_data.empty:
                grouped_data_lomba = filtered_data.groupby(['Prediction', 'Pengalaman Lomba']).size().unstack(fill_value=0)
                grouped_data_lomba.plot(kind='bar', ax=axs[plot_count], stacked=False, color=['#28a745', '#1f77b4'])
                axs[plot_count].set_xlabel('Prediction')
                axs[plot_count].set_ylabel('Count')
                axs[plot_count].set_title('Prediction Results by Pengalaman Lomba')
                axs[plot_count].set_xticklabels(axs[plot_count].get_xticklabels(), rotation=0)
                axs[plot_count].legend(title='Pengalaman Lomba')
                plot_count += 1
            
            # Remove any extra subplots
            if plot_count < len(axs):
                for i in range(plot_count, len(axs)):
                    fig.delaxes(axs[i])
            
            # Adjust layout
            plt.tight_layout()

            # Save the plot to a bytes buffer
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close(fig)
            return send_file(img, mimetype='image/png')
        else:
            return jsonify({"error": "'Prediction' column not found in data"}), 400
    else:
        return jsonify({"error": "No predictions available to plot"}), 400

@app.route('/plot_confusion_matrix')
def plot_confusion_matrix():
    global y_test2, y_test_pred2

    # Check if y_test and y_test_pred are defined
    if y_test2 is None or y_test_pred2 is None:
        return jsonify({"error": "Model not trained yet. Upload a file first."}), 400

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test2, y_test_pred2)
    print("Confusion Matrix:\n", conf_matrix)  # Debugging print statement
    
    # Visualize confusion matrix using heatmap
    plt.figure(figsize=(7, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save plot to BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Clean up plot to prevent memory leaks
    plt.clf()
    plt.close()
    
    # Return the plot as a response
    return send_file(img, mimetype='image/png')

@app.route('/plot_heatmap')
def plot_heatmap():
    global X2, y2
    filter_option = request.args.get('filter', 'cepat')

    # Menggabungkan X dan y menjadi satu DataFrame untuk visualisasi
    df_corr = X2.copy()
    df_corr['Masa Tunggu'] = y2

    # One-Hot Encoding untuk kolom kategorikal
    df_corr = pd.get_dummies(df_corr, drop_first=True)

    if filter_option == 'cepat':
        # Menghitung korelasi untuk 'Masa Tunggu_CEPAT'
        df_corr['Masa Tunggu_CEPAT'] = 1 - df_corr['Masa Tunggu_LAMBAT']
        correlation_matrix = df_corr.corr()
        title = 'Correlation Heatmap for Masa Tunggu CEPAT'
    elif filter_option == 'lambat':
        # Menghitung korelasi untuk 'Masa Tunggu_LAMBAT'
        correlation_matrix = df_corr.corr()
        title = 'Correlation Heatmap for Masa Tunggu LAMBAT'
    else:
        return jsonify({"error": "Invalid filter option"}), 400

    # Membuat heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)

    # Save plot to BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Clean up plot to prevent memory leaks
    plt.clf()
    plt.close()
    
    # Return the plot as a response
    return send_file(img, mimetype='image/png')

@app.route('/download_template_klasifikasi1', methods=['GET'])
def download_template_klasifikasi1():
    # Definisikan struktur data
    data = {
        'SID': [],
        'Gender': [],
        'IPS 1': [],
        'IPS 2': [],
        'IPS 3': [],
        'IPS 4': [],
        'IPS 5': [],
        'IPS 6': []
    }

    # Buat DataFrame
    df = pd.DataFrame(data)

    # Simpan DataFrame ke file Excel di memory buffer
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
    
    # Pindahkan pointer ke awal buffer
    output.seek(0)

    # Kirim file sebagai respons
    return send_file(output, download_name='template_klasifikasi1.xlsx', as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')



@app.route('/download_template_klasifikasi2', methods=['GET'])
def download_template_klasifikasi2():
    # Definisikan struktur data
    data = {
        'NIM': [],
        'Gender': [],
        'Lama Masa Studi': [],
        'IPK': [],
        'Score EPRT': [],
        'Pengalaman Lomba': [],
        'TAK': []
    }

    # Buat DataFrame
    df = pd.DataFrame(data)

    # Simpan DataFrame ke file Excel di memory buffer
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
    
    # Pindahkan pointer ke awal buffer
    output.seek(0)

    # Kirim file sebagai respons
    return send_file(output, download_name='template_klasifikasi2.xlsx', as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    app.run(debug=True)
