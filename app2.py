#random forest

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import emoji
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
from sklearn.ensemble import RandomForestRegressor  
app = Flask(__name__)
CORS(app)


df = pd.read_excel("D:\\SY\\3rd SEM\\FDS\\Final Project\\FDS.xlsx")

def remove_emojis(text):
    if isinstance(text, str):
        return ''.join(c for c in text if c not in emoji.EMOJI_DATA)
    else:
        return text
    
df['reviews'] = df['review'].apply(remove_emojis)
df['review_title'] = df['review_title'].apply(remove_emojis)

df.drop('review',axis=1,inplace=True)

df.rename(columns={'Review_without_emojis': 'reviews'}, inplace=True)

review_title_column_name = 'review_title',

def remove_emojis(text):
  if isinstance(text, str):
    return ''.join(c for c in text if c not in emoji.EMOJI_DATA)
  else:
    return text
  
df['Review_title_without_emojis'] = df['review_title'].apply(remove_emojis)

df.drop('review_title',axis=1,inplace=True)

df.rename(columns={'Review_title_without_emojis': 'review_title'}, inplace=True)

df = df.reindex(columns=['Index','URL', 'review_rating', 'review_title', 'reviews'])

sia = SentimentIntensityAnalyzer()

df['reviews'] = df['reviews'].astype(str) 

res = {}

for i, row in tqdm(df.iterrows()):
    review = row["reviews"]
    index = row["Index"]
    
    if pd.notna(review):
        res[index] = sia.polarity_scores(review)
        
df.rename(columns={'Index': 'index'}, inplace=True)

vaders=pd.DataFrame(res).T
vaders=vaders.reset_index().rename(columns={'Index':'Id'})
merged_df=pd.merge(df,vaders,on='index')
merged_df.drop('URL',axis=1,inplace=True)
merged_df['sentiment'] = np.where(merged_df['compound'] > 0, 'positive', np.where(merged_df['compound'] < 0, 'negative', 'neutral'))


X_train, X_test, y_train, y_test = train_test_split(
    merged_df[['compound', 'reviews', 'review_title']], merged_df['review_rating'], test_size=0.2, random_state=2
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['reviews'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['reviews'])

X_train_final = X_train[['compound']].join(pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()), how='left')
X_test_final = X_test[['compound']].join(pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()), how='left')
X_train_final = X_train_final.fillna(0)
X_test_final = X_test_final.fillna(0)

rf_model = RandomForestRegressor(n_estimators=100, random_state=2)
rf_predictions = rf_model.predict(X_test_final)
rf_rounded_predictions = rf_predictions.round().astype(int)
rf_rounded_predictions = np.clip(rf_rounded_predictions, 1, 5)
rf_mse = mean_squared_error(y_test, rf_rounded_predictions)
print(f"Mean Squared Error (MSE) for Random Forest: {rf_mse}")


with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

with open('tfidf_vectorizer.pkl', 'rb') as model_file:
    tfidf_vectorizer = pickle.load(model_file)

with open('sentiment_analyzer.pkl', 'wb') as file:
    pickle.dump(sia, file)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/firstapi/', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_title = request.form.get('review_title')
        user_review = request.form.get('review')

        user_title = remove_emojis(user_title)
        user_review = remove_emojis(user_review)

        scores = sia.polarity_scores(user_review)

        user_review_tfidf = tfidf_vectorizer.transform([user_review])
                
        user_review_array = user_review_tfidf.toarray()

        features_review = tfidf_vectorizer.get_feature_names_out()
        
        user_data_tfidf = pd.DataFrame({
            'compound': [scores['compound']],
            'reviews': [user_review],
            'review_title': [user_title]
        }).reindex(columns=X_train_final.columns, fill_value=0)

        rating_prediction = rf_model.predict(user_data_tfidf)[0]

        rating_prediction = int(np.clip(round(rating_prediction), 1, 5))
        sentiment = 'positive' if rating_prediction > 3 else 'negative' if rating_prediction < 3 else 'neutral'

        result = {
            'prediction': rating_prediction,
            'sentiment': sentiment
        }

        return jsonify(result)

    return "Hello, this is the GET method for /firstapi/"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=105)