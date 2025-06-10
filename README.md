Emotion Classifier
This project is an Emotion Classifier built using machine learning techniques. It predicts the emotional tone of a given text, categorizing it into emotions like sadness, joy, love, anger, fear, or surprise. The application is developed with Streamlit for an interactive web interface, making it easy for users to input text and get instant emotion predictions.

Link of the User Interface: https://emotion-classifier-4.onrender.com/

🚀 Features
Emotion Prediction: Classifies text into one of six core emotions.
Text Preprocessing: Cleans input text by removing URLs, mentions, and non-alphabetic characters.
TF-IDF Vectorization: Transforms text data into numerical features for model input.
XGBoost Classifier: Utilizes a powerful gradient-boosting model for accurate predictions.
Interactive Web UI: A user-friendly interface built with Streamlit for seamless interaction.
Custom Styling: Incorporates custom CSS to enhance the application's visual appeal, including a background image.
🛠️ Technologies Used
Python: The core programming language.
datasets library: For loading the "dair-ai/emotion" dataset.
pandas: For data manipulation and structuring.
scikit-learn: For TfidfVectorizer, LabelEncoder, and evaluation metrics.
xgboost: For the XGBClassifier model.
joblib: For saving and loading trained models and preprocessors.
streamlit: For creating the interactive web application.
re (Regular Expressions): For text cleaning.
base64: For embedding images directly into Streamlit's custom CSS.
📂 Project Structure
.
├── app.py                # Main Streamlit application and prediction logic
├── xgb_model2.pkl        # Trained XGBoost model
├── vectorizer2.pkl       # Fitted TF-IDF Vectorizer
├── label_encoder2.pkl    # Fitted Label Encoder
├── Mask.webp             # Background image for the Streamlit app
├── requirements.txt      # Python dependencies
└── README.md             # This file
📊 Dataset
The model is trained on the dair-ai/emotion dataset, which consists of English Twitter messages labeled with one of six emotions:

sadness
joy
love
anger
fear
surprise
🧠 Model Training
The emotion classification model follows these steps:

Data Loading: The dair-ai/emotion dataset is loaded and converted to Pandas DataFrames.
Label Mapping: Numeric labels are mapped to human-readable emotion names.
Text Cleaning: A clean_text function is applied to preprocess the raw tweet text, removing unwanted characters, URLs, and mentions.
Feature Extraction: TfidfVectorizer is used to convert the cleaned text into numerical TF-IDF features.
Label Encoding: Emotion labels are transformed into numerical representations using LabelEncoder.
Model Training: An XGBClassifier is trained on the vectorized text data with early stopping to prevent overfitting.
Model Saving: The trained XGBoost model, TfidfVectorizer, and LabelEncoder are saved using joblib for later use in the Streamlit application.
🚀 Deployment on Render
This application is designed for easy deployment on platforms like Render.

Deployment Steps:
Prepare your files: Ensure all necessary files (app.py, xgb_model2.pkl, vectorizer2.pkl, label_encoder2.pkl, Mask.webp, requirements.txt, Procfile) are in the root of your project directory.

Create requirements.txt: This file lists all Python libraries your project depends on.

datasets
pandas
scikit-learn
xgboost
joblib
streamlit
# Add any other libraries you might have used, e.g., numpy
Create Procfile: This file tells Render how to start your Streamlit application. Create a file named Procfile (no extension) in the root of your repository with the following content:

web: streamlit run app.py --server.port $PORT
web: specifies that this is a web service.
streamlit run app.py executes your Streamlit application.
--server.port $PORT is crucial for Render, as it tells Streamlit to bind to the specific port assigned by the hosting environment.
Push to a Git Repository: Upload your entire project to a Git repository (e.g., GitHub, GitLab, Bitbucket).

Deploy on Render:

Go to your Render Dashboard.
Click "New" -> "Web Service".
Connect your Git repository.
Configure the service settings:
Service Name: Give it a descriptive name (e.g., emotion-classifier).
Region: Choose a region close to your target users.
Branch: Select the branch you want to deploy (e.g., main or master).
Root Directory: Leave blank if your code is in the repository root.
Build Command: Render will usually auto-detect pip install -r requirements.txt.
Start Command: Render will automatically use the Procfile command (streamlit run app.py --server.port $PORT). You can leave this blank in the Render UI if your Procfile is correct, or explicitly enter it.
Instance Type: Start with "Free" for testing.
Click "Create Web Service" and monitor the deployment logs.
⚙️ How to Run Locally
To run this application on your local machine:

Clone the repository:
Bash

git clone <https://github.com/Ankuj999/Emotion_Classifier>
cd <Emotion_Classifier>
Create a virtual environment (recommended):
    python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install dependencies:**bash
pip install -r requirements.txt
4. **Run the Streamlit application:**bash
streamlit run app.py
```
This will open the application in your web browser, usually at http://localhost:8501.

🙏 Acknowledgements
dair-ai/emotion dataset: For providing the valuable dataset used for training.
✍️ Developed By
Ankuj Saha Data Analyst and Machine Learning Developer

© 2025 Ankuj. All rights reserved.


