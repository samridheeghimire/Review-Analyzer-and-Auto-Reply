import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = {
    "text": [
        #for positive
        "The food was absolutely delicious and tasty!",
        "the food was great and amazing",
        "Great ambiance and quick service, loved it.",
        "Delicious meal and prompt delivery, 10/10.",
        "Super helpful customer support and friendly staff.",
        "Loved the experience, very polite staff and clean tables.",
        "Excellent price for such great portions.",
        "Fast delivery and hot fresh food!",
        "Best restaurant in town, highly recommended.",
        "Wonderful coffee and cozy environment.",
        
        #for neutral
        "The food was okay, nothing special.",
        "Average experience, standard pricing.",
        "Decent meal but took some time to arrive.",
        "It was fine, ordinary service.",
        "Neither good nor bad, just an everyday spot.",
        "Prices are moderate, food is acceptable.",
        "Standard delivery time, package was intact.",
        "The ambience was fine, food tasted normal.",
        "Decent place to grab a quick bite.",
        "Service was alright, nothing extraordinary.",
        
        #for negative
        "Terrible service, waited 45 minutes for cold food!",
        "The food was bland, tasteless, and completely cold.",
        "Rude staff and extremely overpriced.",
        "Worst dining experience ever, will never return.",
        "Delivery was delayed by two hours and items were missing.",
        "Horrible quality and unhygienic packaging.",
        "Disgusting taste and arrogant manager.",
        "Dirty tables and very slow waiter.",
        "Overpriced bill with hidden extra charges.",
        "My order was completely wrong and ruined."
    ],
    "sentiment": [
        "Positive"] * 10 + ["Neutral"] * 10 + ["Negative"] * 10
}

df = pd.DataFrame(data)

#split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

#pipeline: TF-IDF + Multinomial Naive Bayes
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", MultinomialNB(alpha=0.1))
])

#train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

#overwrite saved model
joblib.dump(pipeline, "sentiment_pipeline.pkl")
