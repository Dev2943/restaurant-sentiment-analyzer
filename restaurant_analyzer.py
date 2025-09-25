#!/usr/bin/env python3
"""
Restaurant Review Sentiment Analyzer & Insights Dashboard
Complete NLP project for analyzing restaurant reviews

Features:
- Sentiment analysis (multiple approaches)
- Topic modeling and theme discovery
- Word cloud generation
- Aspect-based sentiment analysis
- Business insights dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RestaurantReviewAnalyzer:
    """Complete sentiment analysis system for restaurant reviews"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {}
        self.sample_data = self._create_sample_data()
        
    def _create_sample_data(self):
        """Create sample restaurant review data for demonstration"""
        reviews = [
            # Positive reviews
            "Amazing Italian restaurant! The pasta was perfectly cooked and the service was outstanding. Highly recommend the tiramisu!",
            "Best pizza in town! Fresh ingredients, crispy crust, and friendly staff. Will definitely come back.",
            "Fantastic dining experience. The ambiance was perfect for our anniversary dinner. Food was delicious and reasonably priced.",
            "Love this place! Great food, fast service, and clean environment. The chicken parmesan is to die for.",
            "Excellent restaurant with authentic flavors. The chef really knows what they're doing. Perfect for date night.",
            "Outstanding service and incredible food quality. The seafood was fresh and cooked to perfection.",
            "Wonderful atmosphere and delicious food. The staff was attentive and the wine selection was impressive.",
            "Best Thai food I've ever had! Spicy just right, fresh vegetables, and reasonable prices.",
            
            # Negative reviews
            "Terrible experience. Food was cold, service was slow, and the place was dirty. Would not recommend.",
            "Overpriced and underwhelming. The steak was tough and the vegetables were overcooked. Very disappointed.",
            "Poor service and mediocre food. Waited 45 minutes for our order and it wasn't worth it.",
            "Not impressed at all. The pasta was bland, the bread was stale, and the waitress was rude.",
            "Awful restaurant. Food poisoning after eating here. Health department should investigate.",
            "Complete waste of money. Tiny portions, high prices, and tasteless food. Never again.",
            "Horrible experience. Long wait times, cold food, and unfriendly staff. Avoid this place.",
            "Disappointing meal. The pizza was soggy, the salad was wilted, and the service was terrible.",
            
            # Mixed reviews
            "Good food but terrible service. The pizza was great but we waited forever to get it.",
            "Nice atmosphere and decent food, but quite expensive for what you get.",
            "The appetizers were amazing but the main course was just okay. Hit or miss.",
            "Great location and ambiance, food was average. Good for drinks but not for dinner.",
            "Friendly staff and quick service, but the food was nothing special. Average experience.",
            "Beautiful restaurant with mediocre food. Better for the atmosphere than the cuisine."
        ]
        
        # Create corresponding ratings and metadata
        ratings = [5, 5, 4, 5, 4, 5, 4, 5,  # Positive
                  1, 2, 2, 1, 1, 1, 1, 2,   # Negative  
                  3, 3, 3, 3, 3, 3]         # Mixed
        
        restaurants = ['Mario\'s Italian', 'Tony\'s Pizza', 'Fine Dining Co', 'Mama Mia\'s', 'Bistro 21',
                      'Ocean Grill', 'Wine & Dine', 'Thai Garden', 'Burger Joint', 'Steakhouse Prime',
                      'Quick Eats', 'Pasta Corner', 'Food Truck', 'Cheap Eats', 'Fast Food', 'Pizza Hut',
                      'Local Diner', 'City Bistro', 'Corner Cafe', 'Downtown Grill', 'Family Restaurant', 'Rooftop Bar']
        
        cuisines = ['Italian', 'Italian', 'American', 'Italian', 'French', 'Seafood', 'American', 'Thai',
                   'American', 'American', 'Fast Food', 'Italian', 'Street Food', 'Fast Food', 'Fast Food', 'Italian',
                   'American', 'American', 'American', 'American', 'American', 'American']
        
        return pd.DataFrame({
            'review_text': reviews,
            'rating': ratings,
            'restaurant_name': restaurants[:len(reviews)],
            'cuisine_type': cuisines[:len(reviews)]
        })
    
    def load_data(self, file_path=None):
        """Load restaurant review data from file or use sample data"""
        if file_path:
            try:
                return pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"File {file_path} not found. Using sample data instead.")
                return self.sample_data
        else:
            print("Using sample restaurant review data for demonstration.")
            return self.sample_data
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
    
    def train_ml_models(self, df):
        """Train machine learning models for sentiment classification"""
        # Prepare data
        X = df['processed_text']
        y = df['sentiment_label']
        
        # Convert text to TF-IDF vectors
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models = {
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score
            }
            
            print(f"{name}:")
            print(f"  Training Accuracy: {train_score:.3f}")
            print(f"  Testing Accuracy: {test_score:.3f}")
            print()
        
        self.models = results
        return results
    
    def perform_topic_modeling(self, df, n_topics=5):
        """Perform topic modeling using LDA"""
        # Prepare text data
        processed_texts = df['processed_text'].tolist()
        
        # Create document-term matrix
        vectorizer = CountVectorizer(max_features=100, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(processed_texts)
        
        # Perform LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic': topic_idx + 1,
                'words': top_words,
                'weights': topic[top_words_idx]
            })
        
        return topics
    
    def create_word_cloud(self, text_data, title="Word Cloud"):
        """Generate word cloud visualization"""
        combined_text = ' '.join(text_data)
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(combined_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def analyze_aspects(self, df):
        """Perform aspect-based sentiment analysis"""
        aspects = {
            'food': ['food', 'taste', 'flavor', 'delicious', 'meal', 'dish', 'cooking', 'chef'],
            'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'friendly', 'rude'],
            'atmosphere': ['atmosphere', 'ambiance', 'environment', 'music', 'lighting', 'decor'],
            'price': ['price', 'expensive', 'cheap', 'cost', 'value', 'money', 'affordable']
        }
        
        aspect_sentiments = {aspect: [] for aspect in aspects}
        
        for _, row in df.iterrows():
            text = row['review_text'].lower()
            
            for aspect, keywords in aspects.items():
                aspect_mentions = [word for word in keywords if word in text]
                
                if aspect_mentions:
                    sentiment, score = self.analyze_sentiment_vader(text)
                    aspect_sentiments[aspect].append(score)
        
        # Calculate average sentiment per aspect
        aspect_avg = {}
        for aspect, scores in aspect_sentiments.items():
            if scores:
                aspect_avg[aspect] = np.mean(scores)
            else:
                aspect_avg[aspect] = 0
        
        return aspect_avg
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sentiment Distribution
        sentiment_counts = df['sentiment_label'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Sentiment Distribution', fontweight='bold')
        
        # 2. Rating Distribution
        axes[0, 1].hist(df['rating'], bins=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Rating Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Sentiment vs Rating
        sentiment_rating = df.groupby(['sentiment_label', 'rating']).size().unstack(fill_value=0)
        sentiment_rating.plot(kind='bar', ax=axes[0, 2], stacked=True)
        axes[0, 2].set_title('Sentiment by Rating', fontweight='bold')
        axes[0, 2].set_xlabel('Sentiment')
        axes[0, 2].legend(title='Rating', bbox_to_anchor=(1.05, 1))
        
        # 4. Cuisine Type Analysis
        cuisine_sentiment = pd.crosstab(df['cuisine_type'], df['sentiment_label'])
        cuisine_sentiment.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Sentiment by Cuisine Type', fontweight='bold')
        axes[1, 0].set_xlabel('Cuisine Type')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Review Length Analysis
        df['review_length'] = df['review_text'].str.len()
        axes[1, 1].scatter(df['review_length'], df['rating'], alpha=0.6)
        axes[1, 1].set_title('Review Length vs Rating', fontweight='bold')
        axes[1, 1].set_xlabel('Review Length (characters)')
        axes[1, 1].set_ylabel('Rating')
        
        # 6. Sentiment Scores Distribution
        axes[1, 2].hist(df['sentiment_score'], bins=20, alpha=0.7, color='lightcoral')
        axes[1, 2].set_title('Sentiment Scores Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Sentiment Score')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def generate_insights(self, df):
        """Generate business insights from the analysis"""
        insights = []
        
        # Overall sentiment statistics
        total_reviews = len(df)
        positive_pct = (df['sentiment_label'] == 'positive').mean() * 100
        negative_pct = (df['sentiment_label'] == 'negative').mean() * 100
        avg_rating = df['rating'].mean()
        
        insights.append(f"📊 OVERALL PERFORMANCE")
        insights.append(f"   • Total Reviews Analyzed: {total_reviews}")
        insights.append(f"   • Positive Sentiment: {positive_pct:.1f}%")
        insights.append(f"   • Negative Sentiment: {negative_pct:.1f}%")
        insights.append(f"   • Average Rating: {avg_rating:.1f}/5.0")
        insights.append("")
        
        # Best and worst performing restaurants
        restaurant_avg = df.groupby('restaurant_name')['rating'].mean().sort_values(ascending=False)
        insights.append(f"🏆 TOP PERFORMING RESTAURANTS")
        for restaurant in restaurant_avg.head(3).index:
            rating = restaurant_avg[restaurant]
            insights.append(f"   • {restaurant}: {rating:.1f}/5.0")
        insights.append("")
        
        insights.append(f"⚠️  NEEDS IMPROVEMENT")
        for restaurant in restaurant_avg.tail(3).index:
            rating = restaurant_avg[restaurant]
            insights.append(f"   • {restaurant}: {rating:.1f}/5.0")
        insights.append("")
        
        # Cuisine analysis
        cuisine_performance = df.groupby('cuisine_type')['rating'].mean().sort_values(ascending=False)
        insights.append(f"🍽️  CUISINE PERFORMANCE")
        for cuisine in cuisine_performance.index:
            rating = cuisine_performance[cuisine]
            insights.append(f"   • {cuisine}: {rating:.1f}/5.0")
        insights.append("")
        
        # Common positive and negative words
        positive_reviews = df[df['sentiment_label'] == 'positive']['processed_text']
        negative_reviews = df[df['sentiment_label'] == 'negative']['processed_text']
        
        if not positive_reviews.empty:
            pos_words = ' '.join(positive_reviews).split()
            pos_common = pd.Series(pos_words).value_counts().head(5)
            insights.append(f"✅ MOST MENTIONED POSITIVE ASPECTS")
            for word, count in pos_common.items():
                insights.append(f"   • {word}: {count} mentions")
            insights.append("")
        
        if not negative_reviews.empty:
            neg_words = ' '.join(negative_reviews).split()
            neg_common = pd.Series(neg_words).value_counts().head(5)
            insights.append(f"❌ MOST MENTIONED NEGATIVE ASPECTS")
            for word, count in neg_common.items():
                insights.append(f"   • {word}: {count} mentions")
        
        return "\n".join(insights)
    
    def run_complete_analysis(self, file_path=None):
        """Run the complete sentiment analysis pipeline"""
        print("🚀 Starting Restaurant Review Sentiment Analysis")
        print("=" * 50)
        
        # Load data
        df = self.load_data(file_path)
        print(f"✅ Loaded {len(df)} reviews")
        
        # Preprocess text
        df['processed_text'] = df['review_text'].apply(self.preprocess_text)
        print("✅ Preprocessed text data")
        
        # Analyze sentiment using multiple methods
        textblob_results = df['review_text'].apply(lambda x: self.analyze_sentiment_textblob(x))
        vader_results = df['review_text'].apply(lambda x: self.analyze_sentiment_vader(x))
        
        df['sentiment_label'] = textblob_results.apply(lambda x: x[0])
        df['sentiment_score'] = textblob_results.apply(lambda x: x[1])
        df['vader_label'] = vader_results.apply(lambda x: x[0])
        df['vader_score'] = vader_results.apply(lambda x: x[1])
        
        print("✅ Completed sentiment analysis")
        
        # Train ML models
        print("\n🤖 Training Machine Learning Models:")
        print("-" * 30)
        self.train_ml_models(df)
        
        # Topic modeling
        print("🔍 Discovering Topics...")
        topics = self.perform_topic_modeling(df)
        print(f"✅ Discovered {len(topics)} topics")
        
        # Aspect analysis
        print("🎯 Analyzing Aspects...")
        aspects = self.analyze_aspects(df)
        print("✅ Completed aspect analysis")
        
        # Create visualizations
        print("📈 Creating Visualizations...")
        self.create_visualizations(df)
        
        # Word clouds
        print("☁️  Generating Word Clouds...")
        positive_text = df[df['sentiment_label'] == 'positive']['processed_text']
        negative_text = df[df['sentiment_label'] == 'negative']['processed_text']
        
        if not positive_text.empty:
            self.create_word_cloud(positive_text, "Positive Reviews - Word Cloud")
        
        if not negative_text.empty:
            self.create_word_cloud(negative_text, "Negative Reviews - Word Cloud")
        
        # Display topics
        print("\n🏷️  DISCOVERED TOPICS:")
        print("-" * 25)
        for topic in topics:
            print(f"Topic {topic['topic']}: {', '.join(topic['words'][:5])}")
        
        # Display aspect analysis
        print(f"\n🎯 ASPECT-BASED SENTIMENT:")
        print("-" * 30)
        for aspect, score in aspects.items():
            emoji = "😊" if score > 0.1 else "😞" if score < -0.1 else "😐"
            print(f"{emoji} {aspect.title()}: {score:.3f}")
        
        # Generate insights
        print(f"\n💡 BUSINESS INSIGHTS:")
        print("-" * 25)
        insights = self.generate_insights(df)
        print(insights)
        
        print(f"\n🎉 Analysis Complete!")
        print("=" * 50)
        
        return df

# Example usage and demonstration
def main():
    """Main function to demonstrate the restaurant review analyzer"""
    
    # Initialize the analyzer
    analyzer = RestaurantReviewAnalyzer()
    
    # Run complete analysis
    results_df = analyzer.run_complete_analysis()
    
    # Display sample predictions
    print(f"\n🔮 SAMPLE PREDICTIONS:")
    print("-" * 25)
    sample_reviews = [
        "The food was absolutely amazing and the service was perfect!",
        "Terrible experience, cold food and rude staff.",
        "Decent place, nothing special but okay for a quick meal."
    ]
    
    for review in sample_reviews:
        sentiment, score = analyzer.analyze_sentiment_textblob(review)
        print(f"Review: \"{review[:50]}...\"")
        print(f"Predicted Sentiment: {sentiment.title()} (Score: {score:.3f})")
        print()

if __name__ == "__main__":
    main()

"""
NEXT STEPS TO EXTEND THIS PROJECT:

1. 📊 Data Sources:
   - Connect to Yelp API for real restaurant data
   - Scrape TripAdvisor or Google Reviews
   - Use Kaggle restaurant datasets

2. 🧠 Advanced NLP:
   - Implement BERT/RoBERTa for better accuracy
   - Add emotion detection (joy, anger, surprise)
   - Multi-language support

3. 📱 Dashboard Creation:
   - Build Streamlit web app
   - Create interactive Plotly dashboards
   - Add real-time monitoring

4. 🔍 Business Intelligence:
   - Competitive analysis features
   - Trend prediction models
   - Recommendation systems

5. 🚀 Production Features:
   - API endpoints for businesses
   - Automated report generation
   - Alert systems for negative sentiment spikes

Run this script to see the complete sentiment analysis in action!
"""