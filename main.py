from datasets import load_dataset
emotions = load_dataset("dair-ai/emotion")
emotions

import pandas as pd

train_df = emotions['train'].to_pandas()
val_df = emotions['validation'].to_pandas()
test_df = emotions['test'].to_pandas()


label_names = emotions['train'].features['label'].int2str

train_df['emotion'] = train_df['label'].apply(label_names)
val_df['emotion'] = val_df['label'].apply(label_names)
test_df['emotion'] = test_df['label'].apply(label_names)

import re

def clean_text(text):
    return re.sub(r"http\S+|@\w+|[^a-zA-Z\s]", "", text.lower())

train_df['clean_text'] = train_df['text'].apply(clean_text)
val_df['clean_text'] = val_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_val = vectorizer.transform(val_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(train_df['emotion'])
y_val = le.transform(val_df['emotion'])
y_test = le.transform(test_df['emotion'])



from xgboost import XGBClassifier

# Initialize model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)

# Fit with early stopping
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)






from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

import joblib
joblib.dump(model, "xgb_model2.pkl")
joblib.dump(vectorizer, "vectorizer2.pkl")
joblib.dump(le, "label_encoder2.pkl")

