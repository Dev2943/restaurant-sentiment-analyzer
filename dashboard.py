"""
Restaurant Review Sentiment Analyzer - Enhanced Dashboard v2.0
Advanced NLP Pipeline with Real-time Analysis, Multi-Restaurant Comparison, and Business Intelligence

Features:
- Modern, intuitive UI with tabs
- Multiple restaurant comparison
- Real-time sentiment analysis
- Advanced ML models (TextBlob, VADER, Ensemble)
- Topic modeling and insights
- Export to PDF/Excel
- Interactive visualizations
- Demo data with 750+ reviews
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime
import io
import base64
import random

# NLP and ML Libraries
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Restaurant Sentiment Analyzer Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #4ECDC4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'restaurants' not in st.session_state:
    st.session_state.restaurants = []

def generate_demo_data():
    """Generate realistic demo data with 750 reviews"""
    
    # Restaurant names
    restaurants = [
        "Italian Bistro", "Sushi Palace", "Burger Haven", "Thai Garden", "Mexican Cantina",
        "French Quarter", "Indian Spice", "Pizza Paradise", "Steakhouse Prime", "Ocean Grill",
        "Vegan Delight", "BBQ Shack", "Chinese Dynasty", "Greek Taverna", "Tapas Bar",
        "Ramen House", "Seafood Market", "Farm Table", "Wine & Dine", "Breakfast Club"
    ]
    
    # Positive review templates
    positive_reviews = [
        "Absolutely amazing! The {dish} was perfectly cooked and the service was outstanding. Will definitely be back!",
        "Best {cuisine} food I've ever had! Fresh ingredients, great atmosphere, and friendly staff.",
        "Incredible experience! The {dish} exceeded all expectations. Highly recommend this place!",
        "Outstanding service and delicious food. The {dish} was to die for. Five stars!",
        "Loved everything about this restaurant! The {dish} was phenomenal and the ambiance was perfect.",
        "Fantastic meal! The {dish} was cooked to perfection. Great value for the quality.",
        "Excellent dining experience! The {dish} was incredible and the staff was so attentive.",
        "One of the best restaurants in town! The {dish} was absolutely delicious.",
        "Amazing food and wonderful service! The {dish} was the highlight of our evening.",
        "Perfect dinner! The {dish} was exquisite and the wine selection was impressive.",
        "Highly recommend! The {dish} was fresh and flavorful. Great for special occasions.",
        "Superb quality! The {dish} was prepared beautifully and tasted amazing.",
        "Five star experience! The {dish} was incredible and the service impeccable.",
        "Wonderful restaurant! The {dish} was outstanding and the portions were generous.",
        "Best meal we've had in ages! The {dish} was perfectly seasoned and cooked.",
        "Exceptional food and service! The {dish} was a masterpiece. Will return soon!",
        "Absolutely delicious! The {dish} was fresh and the presentation was beautiful.",
        "Great atmosphere and amazing food! The {dish} did not disappoint.",
        "Loved this place! The {dish} was cooked to perfection and staff was friendly.",
        "Top-notch dining! The {dish} was exceptional and the dessert was divine."
    ]
    
    # Negative review templates
    negative_reviews = [
        "Terrible experience. The {dish} was cold and service was extremely slow. Very disappointed.",
        "Not impressed at all. The {dish} was bland and overpriced. Won't be returning.",
        "Poor quality food. The {dish} was undercooked and the place was dirty. Avoid!",
        "Disappointing meal. The {dish} was tough and flavorless. Not worth the money.",
        "Horrible service! Waited forever for our {dish} and it wasn't even good when it arrived.",
        "Very bad experience. The {dish} was stale and the staff was rude. Never again.",
        "Awful restaurant. The {dish} gave me food poisoning. Health department should investigate.",
        "Completely unacceptable. The {dish} was burnt and the vegetables were rotten.",
        "Worst dining experience ever! The {dish} was inedible and service was terrible.",
        "Do not recommend. The {dish} was dry and tasteless. Very overrated place.",
        "Horrible food quality. The {dish} was greasy and cold. Waste of money.",
        "Terrible! The {dish} was undercooked and the place smelled bad. Very unsanitary.",
        "Disappointing all around. The {dish} was nothing special and way too expensive.",
        "Poor service and mediocre food. The {dish} was lukewarm and portions were tiny.",
        "Not good at all. The {dish} was bland and the wait staff was unfriendly.",
        "Very disappointed. The {dish} didn't taste fresh and the ambiance was depressing.",
        "Awful experience! The {dish} was overcooked and rubbery. Never coming back.",
        "Terrible value. The {dish} was small and not flavorful. Too expensive for what you get.",
        "Bad food and worse service. The {dish} was cold when served. Unacceptable!",
        "Don't waste your time. The {dish} was poorly prepared and the place was dirty."
    ]
    
    # Neutral review templates
    neutral_reviews = [
        "The {dish} was okay, nothing special. Service was average. Might try again.",
        "Decent food but nothing to write home about. The {dish} was good but not great.",
        "Average experience overall. The {dish} was fine but expected more for the price.",
        "The {dish} was good but the service could be better. Mixed feelings about this place.",
        "Nice ambiance but the {dish} was just okay. Probably won't rush back.",
        "Acceptable meal. The {dish} was decent but there are better options nearby.",
        "The {dish} was satisfactory. Nothing exceptional but not bad either.",
        "Average restaurant. The {dish} was alright but portions could be bigger.",
        "Fair experience. The {dish} was good but service was slow. Could improve.",
        "The {dish} was decent enough. Not bad but not amazing either. Just average.",
        "Okay place for a quick meal. The {dish} was fine but wouldn't go out of my way.",
        "Mediocre at best. The {dish} was edible but lacked flavor. Could be better.",
        "Nothing special about this place. The {dish} was standard fare. Very average.",
        "The {dish} was acceptable but overpriced. Service was okay, nothing memorable.",
        "Middle of the road restaurant. The {dish} was fine but atmosphere was lacking."
    ]
    
    # Dishes by cuisine type
    dishes = {
        "Italian": ["pasta", "pizza", "risotto", "lasagna", "tiramisu", "carbonara", "bruschetta"],
        "Sushi": ["sushi rolls", "sashimi", "tempura", "ramen", "teriyaki", "miso soup"],
        "American": ["burger", "steak", "fries", "wings", "mac and cheese", "BBQ ribs"],
        "Thai": ["pad thai", "curry", "spring rolls", "tom yum", "satay"],
        "Mexican": ["tacos", "burritos", "enchiladas", "quesadilla", "guacamole", "fajitas"],
        "French": ["escargot", "coq au vin", "ratatouille", "croissant", "crème brûlée"],
        "Indian": ["curry", "naan", "biryani", "tandoori", "samosas", "tikka masala"],
        "Chinese": ["fried rice", "dumplings", "kung pao", "sweet and sour", "spring rolls"],
        "Seafood": ["salmon", "lobster", "shrimp", "fish and chips", "clam chowder", "oysters"],
        "Greek": ["gyro", "souvlaki", "moussaka", "spanakopita", "baklava"]
    }
    
    # Generate reviews
    reviews = []
    restaurant_cuisines = {
        "Italian Bistro": "Italian", "Sushi Palace": "Sushi", "Burger Haven": "American",
        "Thai Garden": "Thai", "Mexican Cantina": "Mexican", "French Quarter": "French",
        "Indian Spice": "Indian", "Pizza Paradise": "Italian", "Steakhouse Prime": "American",
        "Ocean Grill": "Seafood", "Vegan Delight": "American", "BBQ Shack": "American",
        "Chinese Dynasty": "Chinese", "Greek Taverna": "Greek", "Tapas Bar": "Spanish",
        "Ramen House": "Sushi", "Seafood Market": "Seafood", "Farm Table": "American",
        "Wine & Dine": "French", "Breakfast Club": "American"
    }
    
    random.seed(42)  # For reproducibility
    
    for i in range(750):
        restaurant = random.choice(restaurants)
        cuisine = restaurant_cuisines.get(restaurant, "American")
        dish_list = dishes.get(cuisine, dishes["American"])
        dish = random.choice(dish_list)
        
        # 60% positive, 25% negative, 15% neutral (realistic distribution)
        rand = random.random()
        if rand < 0.60:
            template = random.choice(positive_reviews)
        elif rand < 0.85:
            template = random.choice(negative_reviews)
        else:
            template = random.choice(neutral_reviews)
        
        review = template.format(dish=dish, cuisine=cuisine)
        reviews.append({
            'Review': review,
            'Restaurant': restaurant
        })
    
    return pd.DataFrame(reviews)

class SentimentAnalyzer:
    """Enhanced sentiment analysis with multiple models"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def analyze_textblob(self, text):
        """TextBlob sentiment analysis"""
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positive', polarity
            elif polarity < -0.1:
                return 'Negative', polarity
            else:
                return 'Neutral', polarity
        except:
            return 'Neutral', 0.0
    
    def analyze_vader(self, text):
        """VADER sentiment analysis"""
        try:
            scores = self.vader.polarity_scores(str(text))
            compound = scores['compound']
            
            if compound >= 0.05:
                return 'Positive', compound
            elif compound <= -0.05:
                return 'Negative', compound
            else:
                return 'Neutral', compound
        except:
            return 'Neutral', 0.0
    
    def ensemble_analysis(self, text):
        """Ensemble method combining multiple models"""
        tb_sentiment, tb_score = self.analyze_textblob(text)
        vader_sentiment, vader_score = self.analyze_vader(text)
        
        # Weighted average (VADER is often more accurate for short texts)
        ensemble_score = (tb_score * 0.4 + vader_score * 0.6)
        
        if ensemble_score > 0.1:
            sentiment = 'Positive'
        elif ensemble_score < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return sentiment, ensemble_score, tb_sentiment, vader_sentiment

class TopicModeler:
    """Topic modeling for review insights"""
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=1000
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )
    
    def extract_topics(self, texts):
        """Extract main topics from reviews"""
        try:
            doc_term_matrix = self.vectorizer.fit_transform(texts)
            self.lda.fit(doc_term_matrix)
            
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:]]
                topics.append({
                    'topic_id': topic_idx + 1,
                    'keywords': ', '.join(top_words[:5])
                })
            
            return pd.DataFrame(topics)
        except:
            return pd.DataFrame()

def load_data(uploaded_file=None):
    """Load restaurant review data"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def analyze_reviews(df, text_column='Review', restaurant_column=None):
    """Perform comprehensive sentiment analysis"""
    analyzer = SentimentAnalyzer()
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, text in enumerate(df[text_column]):
        sentiment, score, tb_sent, vader_sent = analyzer.ensemble_analysis(text)
        
        result = {
            'Review': text,
            'Sentiment': sentiment,
            'Confidence_Score': abs(score),
            'Ensemble_Score': score,
            'TextBlob_Sentiment': tb_sent,
            'VADER_Sentiment': vader_sent
        }
        
        if restaurant_column and restaurant_column in df.columns:
            result['Restaurant'] = df[restaurant_column].iloc[idx]
        
        results.append(result)
        
        # Update progress
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing... {idx + 1}/{len(df)} reviews")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def create_sentiment_distribution(df):
    """Create sentiment distribution visualization"""
    sentiment_counts = df['Sentiment'].value_counts()
    
    colors = {
        'Positive': '#52c41a',
        'Negative': '#ff4d4f',
        'Neutral': '#faad14'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=[colors.get(s, '#1890ff') for s in sentiment_counts.index],
            text=sentiment_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_sentiment_pie(df):
    """Create sentiment pie chart"""
    sentiment_counts = df['Sentiment'].value_counts()
    
    colors = ['#52c41a', '#ff4d4f', '#faad14']
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="Sentiment Breakdown",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_confidence_distribution(df):
    """Create confidence score distribution"""
    fig = px.histogram(
        df,
        x='Confidence_Score',
        nbins=30,
        color='Sentiment',
        color_discrete_map={
            'Positive': '#52c41a',
            'Negative': '#ff4d4f',
            'Neutral': '#faad14'
        },
        title="Sentiment Confidence Distribution"
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_restaurant_comparison(df):
    """Compare sentiment across restaurants"""
    if 'Restaurant' not in df.columns:
        return None
    
    restaurant_sentiment = df.groupby(['Restaurant', 'Sentiment']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    colors = {'Positive': '#52c41a', 'Negative': '#ff4d4f', 'Neutral': '#faad14'}
    
    for sentiment in restaurant_sentiment.columns:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=restaurant_sentiment.index,
            y=restaurant_sentiment[sentiment],
            marker_color=colors.get(sentiment, '#1890ff')
        ))
    
    fig.update_layout(
        title="Restaurant Sentiment Comparison",
        xaxis_title="Restaurant",
        yaxis_title="Number of Reviews",
        barmode='group',
        template="plotly_white",
        height=500
    )
    
    return fig

def generate_wordcloud(text_data, sentiment=None):
    """Generate word cloud"""
    if sentiment:
        text = ' '.join(text_data[text_data['Sentiment'] == sentiment]['Review'])
        title = f"{sentiment} Reviews Word Cloud"
    else:
        text = ' '.join(text_data['Review'])
        title = "All Reviews Word Cloud"
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

def export_to_excel(df):
    """Export results to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Add summary sheet
        summary = pd.DataFrame({
            'Metric': ['Total Reviews', 'Positive', 'Negative', 'Neutral', 'Avg Confidence'],
            'Value': [
                len(df),
                len(df[df['Sentiment'] == 'Positive']),
                len(df[df['Sentiment'] == 'Negative']),
                len(df[df['Sentiment'] == 'Neutral']),
                f"{df['Confidence_Score'].mean():.2f}"
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Restaurant Sentiment Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced NLP Pipeline with Real-time Analysis & Business Intelligence**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        st.markdown("### 🚀 Quick Start")
        
        # Demo button with 750 reviews
        if st.button("🎬 Try Demo (750 Reviews)", type="primary", use_container_width=True):
            with st.spinner("Generating demo data..."):
                demo_df = generate_demo_data()
                st.session_state.demo_data = demo_df
                st.success(f"✅ Demo data loaded! {len(demo_df)} reviews across 20 restaurants. Scroll down and click 'Analyze Reviews'.")
        
        st.markdown("---")
        st.markdown("### 📤 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload restaurant reviews (CSV)",
            type=['csv'],
            help="CSV file should have a 'Review' column and optionally a 'Restaurant' column"
        )
        
        # Download sample template
        st.markdown("---")
        st.markdown("### 📥 Need a Template?")
        
        sample_template = pd.DataFrame({
            'Review': [
                'Amazing food and great service!',
                'Terrible experience, would not recommend.',
                'Average meal, nothing special.'
            ],
            'Restaurant': ['Restaurant A', 'Restaurant B', 'Restaurant C']
        })
        
        csv_template = sample_template.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Sample CSV",
            data=csv_template,
            file_name="sample_reviews_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### Analysis Settings")
        
        use_ensemble = st.checkbox("Use Ensemble Model", value=True, help="Combines TextBlob + VADER for better accuracy")
        show_topics = st.checkbox("Extract Topics", value=True, help="Identify main themes in reviews")
        
        st.markdown("---")
        st.markdown("### 📖 Quick Guide")
        st.info("""
        **1.** Try demo or upload CSV
        **2.** Click 'Analyze Reviews'
        **3.** Explore insights in tabs
        **4.** Export results
        """)
    
    # Main content
    # Check for demo data or uploaded file
    if 'demo_data' in st.session_state:
        df = st.session_state.demo_data
        st.info(f"📊 Using demo data ({len(df)} reviews). Upload your own CSV to analyze your reviews!")
    elif uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = None
    
    if df is not None:
        st.success(f"✅ Loaded {len(df)} reviews!")
        
        # Show preview
        with st.expander("📋 Preview Data"):
            st.dataframe(df.head(10))
        
        # Detect columns
        text_column = st.selectbox("Select Review Text Column", df.columns, index=0)
        
        restaurant_column = None
        if len(df.columns) > 1:
            has_restaurant = st.checkbox("Data has multiple restaurants?", value=True if 'Restaurant' in df.columns else False)
            if has_restaurant:
                default_idx = list(df.columns).index('Restaurant') + 1 if 'Restaurant' in df.columns else 0
                restaurant_column = st.selectbox("Select Restaurant Column", [None] + list(df.columns), index=default_idx)
        
        # Analyze button
        if st.button("🚀 Analyze Reviews", type="primary"):
            with st.spinner("Analyzing reviews... This may take a moment."):
                results_df = analyze_reviews(df, text_column, restaurant_column)
                st.session_state.analyzed_data = results_df
                st.success("✅ Analysis complete!")
        
        # Display results
        if st.session_state.analyzed_data is not None:
            results_df = st.session_state.analyzed_data
            
            # Key Metrics
            st.markdown("## 📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Reviews", len(results_df))
            
            with col2:
                positive_pct = (len(results_df[results_df['Sentiment'] == 'Positive']) / len(results_df) * 100)
                st.metric("Positive", f"{positive_pct:.1f}%", delta=f"{len(results_df[results_df['Sentiment'] == 'Positive'])} reviews")
            
            with col3:
                negative_pct = (len(results_df[results_df['Sentiment'] == 'Negative']) / len(results_df) * 100)
                st.metric("Negative", f"{negative_pct:.1f}%", delta=f"-{len(results_df[results_df['Sentiment'] == 'Negative'])} reviews", delta_color="inverse")
            
            with col4:
                avg_confidence = results_df['Confidence_Score'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            st.markdown("---")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview",
                "🔍 Detailed Analysis",
                "🏪 Restaurant Comparison",
                "💬 Topics & Insights",
                "📥 Export"
            ])
            
            with tab1:
                st.markdown("### Sentiment Analysis Overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_sentiment_distribution(results_df), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_sentiment_pie(results_df), use_container_width=True)
                
                st.plotly_chart(create_confidence_distribution(results_df), use_container_width=True)
            
            with tab2:
                st.markdown("### Detailed Review Analysis")
                
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment",
                    options=['Positive', 'Negative', 'Neutral'],
                    default=['Positive', 'Negative', 'Neutral']
                )
                
                filtered_df = results_df[results_df['Sentiment'].isin(sentiment_filter)]
                
                st.dataframe(
                    filtered_df[['Review', 'Sentiment', 'Confidence_Score', 'TextBlob_Sentiment', 'VADER_Sentiment']],
                    use_container_width=True
                )
                
                # Sample reviews
                st.markdown("#### 📝 Sample Reviews")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🟢 Most Positive**")
                    top_positive = results_df[results_df['Sentiment'] == 'Positive'].nlargest(3, 'Confidence_Score')
                    for _, row in top_positive.iterrows():
                        st.success(f"⭐ {row['Review'][:150]}...")
                
                with col2:
                    st.markdown("**🔴 Most Negative**")
                    top_negative = results_df[results_df['Sentiment'] == 'Negative'].nsmallest(3, 'Ensemble_Score')
                    for _, row in top_negative.iterrows():
                        st.error(f"❌ {row['Review'][:150]}...")
                
                with col3:
                    st.markdown("**🟡 Neutral Reviews**")
                    neutral = results_df[results_df['Sentiment'] == 'Neutral'].head(3)
                    for _, row in neutral.iterrows():
                        st.info(f"➖ {row['Review'][:150]}...")
            
            with tab3:
                if restaurant_column and 'Restaurant' in results_df.columns:
                    st.markdown("### Restaurant Performance Comparison")
                    
                    fig = create_restaurant_comparison(results_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Restaurant rankings
                    st.markdown("#### 🏆 Restaurant Rankings")
                    
                    restaurant_stats = results_df.groupby('Restaurant').agg({
                        'Sentiment': lambda x: (x == 'Positive').sum() / len(x) * 100,
                        'Confidence_Score': 'mean',
                        'Review': 'count'
                    }).round(2)
                    
                    restaurant_stats.columns = ['Positive %', 'Avg Confidence', 'Total Reviews']
                    restaurant_stats = restaurant_stats.sort_values('Positive %', ascending=False)
                    
                    st.dataframe(restaurant_stats, use_container_width=True)
                else:
                    st.info("Upload data with multiple restaurants to see comparison")
            
            with tab4:
                st.markdown("### Topics & Insights")
                
                if show_topics and len(results_df) >= 10:
                    with st.spinner("Extracting topics..."):
                        topic_modeler = TopicModeler(n_topics=5)
                        topics_df = topic_modeler.extract_topics(results_df['Review'])
                        
                        if not topics_df.empty:
                            st.markdown("#### 🎯 Main Topics Discussed")
                            st.dataframe(topics_df, use_container_width=True)
                
                # Word clouds
                st.markdown("#### ☁️ Word Clouds")
                
                cloud_option = st.radio(
                    "Select Word Cloud",
                    ["All Reviews", "Positive Only", "Negative Only"]
                )
                
                if cloud_option == "All Reviews":
                    fig = generate_wordcloud(results_df)
                elif cloud_option == "Positive Only":
                    fig = generate_wordcloud(results_df, sentiment='Positive')
                else:
                    fig = generate_wordcloud(results_df, sentiment='Negative')
                
                st.pyplot(fig)
                
                # Business insights
                st.markdown("#### 💡 Automated Business Insights")
                
                positive_pct = len(results_df[results_df['Sentiment'] == 'Positive']) / len(results_df) * 100
                
                insights = []
                
                if positive_pct > 70:
                    insights.append("✅ **Strong positive sentiment** - Customers are very satisfied!")
                elif positive_pct < 40:
                    insights.append("⚠️ **Concerning negative sentiment** - Immediate attention needed!")
                else:
                    insights.append("📊 **Mixed sentiment** - Room for improvement.")
                
                avg_confidence = results_df['Confidence_Score'].mean()
                if avg_confidence > 0.7:
                    insights.append("🎯 **High confidence scores** - Sentiment predictions are reliable.")
                
                for insight in insights:
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            with tab5:
                st.markdown("### Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    excel_data = export_to_excel(results_df)
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    else:
        # Welcome screen with better instructions
        st.markdown("## 👋 Welcome to Restaurant Sentiment Analyzer Pro!")
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎬 Try Demo
            Click **"Try Demo (750 Reviews)"** in the sidebar to see the analyzer in action with sample reviews from 20 restaurants!
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Upload Data
            Have your own reviews? Upload a CSV file to analyze your restaurant's customer feedback!
            """)
        
        with col3:
            st.markdown("""
            ### 📥 Get Template
            Don't have a CSV? Download our sample template and add your reviews!
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🚀 Features
        
        - **Advanced NLP Models**: TextBlob, VADER, and Ensemble methods
        - **Multi-Restaurant Comparison**: Compare sentiment across different locations
        - **Topic Modeling**: Automatically identify key themes in reviews
        - **Real-time Analysis**: Process hundreds of reviews in seconds
        - **Interactive Visualizations**: Beautiful charts and word clouds
        - **Export Options**: Download results as CSV or Excel
        
        ### 📊 Demo Dataset
        
        Our demo includes:
        - **750 realistic restaurant reviews**
        - **20 different restaurants**
        - **Multiple cuisine types** (Italian, Sushi, Thai, Mexican, and more)
        - **Varied sentiment distribution** (realistic mix of positive, negative, and neutral)
        
        ### 📋 CSV Format
        
        Your CSV should have:
        - A **"Review"** column with review text
        - Optionally: A **"Restaurant"** column for multi-location comparison
        
        **Example:**
        ```
        Restaurant,Review
        "Italian Bistro","Amazing pasta! Best I've ever had."
        "Sushi Palace","Fresh fish but service was slow."
        ```
        
        ### 🎯 Perfect For
        
        - 🏪 Restaurant owners monitoring customer satisfaction
        - 📊 Business analysts tracking sentiment trends
        - 💼 Marketing teams understanding customer feedback
        - 🎓 Students learning about NLP and sentiment analysis
        
        ### 🚀 Get Started Now!
        
        Click the **"Try Demo (750 Reviews)"** button in the sidebar to see it in action! →
        """)

if __name__ == "__main__":
    main()