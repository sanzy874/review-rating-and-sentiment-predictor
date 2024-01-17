#linear regression

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import emoji
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm

app = Flask(__name__)
CORS(app)

with open('sentiment_analyzer.pkl', 'rb') as file:
    sia = pickle.load(file)

with open('reg_model.pkl', 'rb') as model_file:
    reg_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

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
    merged_df[['compound', 'reviews', 'review_title']], df['review_rating'], test_size=0.2, random_state=2
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['reviews'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['reviews'])

X_train_final = X_train[['compound']].join(pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()), how='left')
X_test_final = X_test[['compound']].join(pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()), how='left')
X_train_final = X_train_final.fillna(0)
X_test_final = X_test_final.fillna(0)

reg_model = LinearRegression()
reg_model.fit(X_train_final, y_train)
reg_predictions = reg_model.predict(X_test_final)
pred=reg_model.predict(X_test_final)
rounded_predictions = pred.round().astype(int)
rounded_predictions = np.clip(rounded_predictions, 1, 5)
mse = mean_squared_error(y_test, rounded_predictions)
print(f"Mean Squared Error (MSE) for Linear Regression model: {mse}")


with open('reg_model.pkl', 'wb') as model_file:
    pickle.dump(reg_model, model_file)

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

        rating_prediction = reg_model.predict(user_data_tfidf)[0]

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





























# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# import pickle
# import emoji
# from nltk.sentiment import SentimentIntensityAnalyzer
# import pandas as pd
# import numpy as np

# app = Flask(__name__)
# CORS(app)

# with open('sentiment_analyzer.pkl', 'rb') as file:
#     sia = pickle.load(file)

# with open('reg_model.pkl', 'rb') as model_file:
#     reg_model = pickle.load(model_file)

# with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
#     tfidf_vectorizer = pickle.load(vectorizer_file)

# @app.route('/firstapi/', methods=['POST'])
# def predict():
#     print("Received a POST request")
#     data = request.get_json()
#     print("Received data:", data)

#     user_title = data.get('review_title', '')
#     user_review = data.get('review', '')

#     def remove_emojis(text):
#         if isinstance(text, str):
#             return ''.join(c for c in text if c not in emoji.EMOJI_DATA)
#         else:
#             return text

#     user_title = remove_emojis(user_title)
#     user_review = remove_emojis(user_review)

#     scores = sia.polarity_scores(user_review)

#     user_review_tfidf = tfidf_vectorizer.transform([user_review])
#     user_title_tfidf = tfidf_vectorizer.transform([user_title])

#     user_data_tfidf = pd.DataFrame({
#         'compound': [scores['compound']],
#         'reviews': [user_review],
#         'review_title': [user_title]
#     }).join(pd.DataFrame(user_review_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()), how='left')

#     user_data_tfidf[['reviews', 'review_title']] = pd.DataFrame(
#         np.concatenate([user_review_tfidf.toarray(), user_title_tfidf.toarray()], axis=1),
#         columns=tfidf_vectorizer.get_feature_names_out()[1:3]
#     )

#     rating_prediction = reg_model.predict(user_data_tfidf)[0]
#     rating_prediction = int(np.clip(round(rating_prediction), 1, 5))
#     sentiment = 'positive' if rating_prediction > 3 else 'negative' if rating_prediction < 3 else 'neutral'

#     result = {
#         'prediction': rating_prediction,
#         'sentiment': sentiment
#     }

#     print("Sending response:", result)
#     return jsonify(result)

# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=105)
