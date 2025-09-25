"""
Interactive Web Dashboard for Restaurant Review Sentiment Analysis
Built with Streamlit for easy deployment and sharing
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
from io import BytesIO
import base64

# Import our custom analyzer
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Set page configuration
st.set_page_config(
    page_title="Restaurant Review Sentiment Analyzer", 
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class RestaurantDashboard:
    """Main dashboard class for displaying restaurant review analytics"""
    
    def __init__(self):
        self.df = None
        
    def load_data(self):
        """Load data with multiple options"""
        st.sidebar.header("📊 Data Source")
        
        data_option = st.sidebar.selectbox(
            "Choose data source:",
            ["Upload CSV File", "Use Sample Data", "Load from Collected Data"]
        )
        
        if data_option == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader(
                "Upload your restaurant reviews CSV", 
                type=['csv']
            )
            if uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(self.df)} reviews")
                
        elif data_option == "Use Sample Data":
            self.df = self.create_sample_data()
            st.sidebar.success("✅ Using sample data for demonstration")
            
        elif data_option == "Load from Collected Data":
            try:
                self.df = pd.read_csv('collected_restaurant_reviews.csv')
                st.sidebar.success(f"✅ Loaded {len(self.df)} collected reviews")
            except FileNotFoundError:
                st.sidebar.error("❌ No collected data found. Run data collection first!")
                self.df = self.create_sample_data()
        
        return self.df is not None
    
    def create_sample_data(self):
        """Create enhanced sample data for demonstration"""
        reviews_data = {
            'review_text': [
                "Amazing Italian restaurant! The pasta was perfectly cooked and the service was outstanding. Highly recommend the tiramisu!",
                "Best pizza in town! Fresh ingredients, crispy crust, and friendly staff. Will definitely come back.",
                "Fantastic dining experience. The ambiance was perfect for our anniversary dinner. Food was delicious and reasonably priced.",
                "Love this place! Great food, fast service, and clean environment. The chicken parmesan is to die for.",
                "Excellent restaurant with authentic flavors. The chef really knows what they're doing. Perfect for date night.",
                "Outstanding service and incredible food quality. The seafood was fresh and cooked to perfection.",
                "Wonderful atmosphere and delicious food. The staff was attentive and the wine selection was impressive.",
                "Best Thai food I've ever had! Spicy just right, fresh vegetables, and reasonable prices.",
                "Terrible experience. Food was cold, service was slow, and the place was dirty. Would not recommend.",
                "Overpriced and underwhelming. The steak was tough and the vegetables were overcooked. Very disappointed.",
                "Poor service and mediocre food. Waited 45 minutes for our order and it wasn't worth it.",
                "Not impressed at all. The pasta was bland, the bread was stale, and the waitress was rude.",
                "Awful restaurant. Food poisoning after eating here. Health department should investigate.",
                "Complete waste of money. Tiny portions, high prices, and tasteless food. Never again.",
                "Horrible experience. Long wait times, cold food, and unfriendly staff. Avoid this place.",
                "Disappointing meal. The pizza was soggy, the salad was wilted, and the service was terrible.",
                "Good food but terrible service. The pizza was great but we waited forever to get it.",
                "Nice atmosphere and decent food, but quite expensive for what you get.",
                "The appetizers were amazing but the main course was just okay. Hit or miss.",
                "Great location and ambiance, food was average. Good for drinks but not for dinner.",
                "Friendly staff and quick service, but the food was nothing special. Average experience.",
                "Beautiful restaurant with mediocre food. Better for the atmosphere than the cuisine."
            ],
            'rating': [5, 5, 4, 5, 4, 5, 4, 5, 1, 2, 2, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3],
            'restaurant_name': [
                'Mario\'s Italian', 'Tony\'s Pizza', 'Fine Dining Co', 'Mama Mia\'s', 'Bistro 21',
                'Ocean Grill', 'Wine & Dine', 'Thai Garden', 'Burger Joint', 'Steakhouse Prime',
                'Quick Eats', 'Pasta Corner', 'Food Truck', 'Cheap Eats', 'Fast Food', 'Pizza Hut',
                'Local Diner', 'City Bistro', 'Corner Cafe', 'Downtown Grill', 'Family Restaurant', 'Rooftop Bar'
            ],
            'cuisine_type': [
                'Italian', 'Italian', 'American', 'Italian', 'French', 'Seafood', 'American', 'Thai',
                'American', 'American', 'Fast Food', 'Italian', 'Street Food', 'Fast Food', 'Fast Food', 'Italian',
                'American', 'American', 'American', 'American', 'American', 'American'
            ]
        }
        
        return pd.DataFrame(reviews_data)
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis function"""
        from textblob import TextBlob
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'Positive', polarity
        elif polarity < -0.1:
            return 'Negative', polarity
        else:
            return 'Neutral', polarity
    
    def process_data(self):
        """Process the loaded data for analysis"""
        if self.df is None:
            return
        
        # Add sentiment analysis
        sentiment_results = self.df['review_text'].apply(self.analyze_sentiment)
        self.df['sentiment'] = sentiment_results.apply(lambda x: x[0])
        self.df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
        
        # Add review length
        self.df['review_length'] = self.df['review_text'].str.len()
        
        # Add review word count
        self.df['word_count'] = self.df['review_text'].str.split().str.len()
    
    def show_overview(self):
        """Display overview metrics and statistics"""
        st.markdown('<h1 class="main-header">🍽️ Restaurant Review Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        if self.df is None:
            st.error("Please load data first!")
            return
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📊 Total Reviews",
                value=f"{len(self.df):,}",
                delta=f"{len(self.df)} new"
            )
        
        with col2:
            avg_rating = self.df['rating'].mean()
            st.metric(
                label="⭐ Avg Rating", 
                value=f"{avg_rating:.1f}/5.0",
                delta=f"{avg_rating - 3:.1f} vs baseline"
            )
        
        with col3:
            positive_pct = (self.df['sentiment'] == 'Positive').mean() * 100
            st.metric(
                label="😊 Positive Sentiment",
                value=f"{positive_pct:.1f}%",
                delta=f"{positive_pct - 50:.1f}%"
            )
        
        with col4:
            negative_pct = (self.df['sentiment'] == 'Negative').mean() * 100
            st.metric(
                label="😞 Negative Sentiment",
                value=f"{negative_pct:.1f}%", 
                delta=f"{negative_pct - 25:.1f}%" if negative_pct < 25 else f"+{negative_pct - 25:.1f}%"
            )
        
        with col5:
            unique_restaurants = self.df['restaurant_name'].nunique()
            st.metric(
                label="🏪 Restaurants",
                value=unique_restaurants,
                delta=f"{unique_restaurants} analyzed"
            )
    
    def show_sentiment_analysis(self):
        """Display detailed sentiment analysis"""
        st.header("💭 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = self.df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={
                    'Positive': '#2E8B57',
                    'Negative': '#DC143C', 
                    'Neutral': '#FFD700'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig_hist = px.histogram(
                self.df, 
                x='rating',
                nbins=5,
                title="Rating Distribution",
                color_discrete_sequence=['#FF6B6B']
            )
            fig_hist.update_layout(
                xaxis_title="Rating",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Sentiment by rating heatmap
        st.subheader("🔥 Sentiment vs Rating Analysis")
        
        sentiment_rating_crosstab = pd.crosstab(self.df['sentiment'], self.df['rating'])
        
        fig_heatmap = px.imshow(
            sentiment_rating_crosstab.values,
            labels=dict(x="Rating", y="Sentiment", color="Count"),
            x=sentiment_rating_crosstab.columns,
            y=sentiment_rating_crosstab.index,
            title="Sentiment vs Rating Heatmap",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def show_restaurant_analysis(self):
        """Display restaurant-specific analysis"""
        st.header("🏪 Restaurant Performance Analysis")
        
        # Restaurant performance metrics
        restaurant_stats = self.df.groupby('restaurant_name').agg({
            'rating': ['mean', 'count'],
            'sentiment_score': 'mean'
        }).round(2)
        
        restaurant_stats.columns = ['avg_rating', 'review_count', 'avg_sentiment']
        restaurant_stats = restaurant_stats.reset_index()
        restaurant_stats = restaurant_stats.sort_values('avg_rating', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Restaurants")
            top_restaurants = restaurant_stats.head(10)
            
            fig_bar = px.bar(
                top_restaurants,
                x='avg_rating',
                y='restaurant_name',
                orientation='h',
                title="Average Rating by Restaurant",
                color='avg_rating',
                color_continuous_scale='RdYlGn'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("📊 Review Count vs Rating")
            
            # Fix for negative sentiment values in size parameter
            restaurant_stats['size_metric'] = restaurant_stats['avg_sentiment'] + 1  # Make all values positive
            fig_scatter = px.scatter(
                restaurant_stats,
                x='review_count',
                y='avg_rating',
                size='size_metric',
                hover_name='restaurant_name',
                title="Review Volume vs Rating",
                color='avg_sentiment',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Cuisine analysis
        st.subheader("🍜 Cuisine Type Performance")
        
        cuisine_stats = self.df.groupby('cuisine_type').agg({
            'rating': 'mean',
            'sentiment': lambda x: (x == 'Positive').mean() * 100
        }).round(2)
        
        cuisine_stats.columns = ['avg_rating', 'positive_sentiment_pct']
        cuisine_stats = cuisine_stats.reset_index()
        
        fig_cuisine = px.scatter(
            cuisine_stats,
            x='avg_rating',
            y='positive_sentiment_pct',
            size='avg_rating',
            hover_name='cuisine_type',
            title="Cuisine Performance: Rating vs Positive Sentiment %",
            color='avg_rating',
            color_continuous_scale='Viridis'
        )
        fig_cuisine.update_layout(
            xaxis_title="Average Rating",
            yaxis_title="Positive Sentiment %"
        )
        st.plotly_chart(fig_cuisine, use_container_width=True)
    
    def show_word_analysis(self):
        """Display word cloud and text analysis"""
        st.header("☁️ Word Cloud Analysis")
        
        # Split by sentiment
        positive_reviews = self.df[self.df['sentiment'] == 'Positive']['review_text']
        negative_reviews = self.df[self.df['sentiment'] == 'Negative']['review_text']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not positive_reviews.empty:
                st.subheader("😊 Positive Reviews Word Cloud")
                positive_text = ' '.join(positive_reviews)
                
                wordcloud_pos = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='Greens',
                    max_words=50
                ).generate(positive_text)
                
                # Convert to image for Streamlit
                img_buffer = BytesIO()
                plt.figure(figsize=(8, 6))
                plt.imshow(wordcloud_pos, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                plt.close()
                
                st.image(img_buffer.getvalue())
        
        with col2:
            if not negative_reviews.empty:
                st.subheader("😞 Negative Reviews Word Cloud")
                negative_text = ' '.join(negative_reviews)
                
                wordcloud_neg = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='Reds',
                    max_words=50
                ).generate(negative_text)
                
                # Convert to image for Streamlit
                img_buffer = BytesIO()
                plt.figure(figsize=(8, 6))
                plt.imshow(wordcloud_neg, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                plt.close()
                
                st.image(img_buffer.getvalue())
        
        # Most common words analysis
        st.subheader("📝 Most Common Words by Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not positive_reviews.empty:
                st.write("**Top Positive Words:**")
                pos_words = ' '.join(positive_reviews).lower().split()
                pos_word_freq = pd.Series(pos_words).value_counts().head(10)
                
                fig_pos_words = px.bar(
                    x=pos_word_freq.values,
                    y=pos_word_freq.index,
                    orientation='h',
                    title="Most Common Words in Positive Reviews",
                    color_discrete_sequence=['#2E8B57']
                )
                st.plotly_chart(fig_pos_words, use_container_width=True)
        
        with col2:
            if not negative_reviews.empty:
                st.write("**Top Negative Words:**")
                neg_words = ' '.join(negative_reviews).lower().split()
                neg_word_freq = pd.Series(neg_words).value_counts().head(10)
                
                fig_neg_words = px.bar(
                    x=neg_word_freq.values,
                    y=neg_word_freq.index,
                    orientation='h',
                    title="Most Common Words in Negative Reviews", 
                    color_discrete_sequence=['#DC143C']
                )
                st.plotly_chart(fig_neg_words, use_container_width=True)
    
    def show_insights(self):
        """Display business insights and recommendations"""
        st.header("💡 Business Insights & Recommendations")
        
        # Calculate key insights
        insights = []
        
        # Overall performance
        avg_rating = self.df['rating'].mean()
        positive_pct = (self.df['sentiment'] == 'Positive').mean() * 100
        
        if avg_rating >= 4.0:
            insights.append("🎉 **Excellent Overall Performance!** Your average rating is above 4.0 stars.")
        elif avg_rating >= 3.5:
            insights.append("✅ **Good Performance.** Your average rating is solid but has room for improvement.")
        else:
            insights.append("⚠️ **Performance Needs Attention.** Your average rating is below 3.5 stars.")
        
        # Sentiment insights
        if positive_pct >= 70:
            insights.append(f"😊 **Strong Customer Satisfaction!** {positive_pct:.1f}% of reviews are positive.")
        elif positive_pct >= 50:
            insights.append(f"📊 **Moderate Satisfaction.** {positive_pct:.1f}% positive sentiment suggests mixed experiences.")
        else:
            insights.append(f"🚨 **Customer Satisfaction Critical.** Only {positive_pct:.1f}% positive sentiment.")
        
        # Restaurant-specific insights
        restaurant_performance = self.df.groupby('restaurant_name')['rating'].mean().sort_values(ascending=False)
        best_restaurant = restaurant_performance.index[0]
        worst_restaurant = restaurant_performance.index[-1]
        
        insights.append(f"🏆 **Top Performer:** {best_restaurant} ({restaurant_performance.iloc[0]:.1f} stars)")
        insights.append(f"📉 **Needs Improvement:** {worst_restaurant} ({restaurant_performance.iloc[-1]:.1f} stars)")
        
        # Cuisine insights
        cuisine_performance = self.df.groupby('cuisine_type')['rating'].mean().sort_values(ascending=False)
        best_cuisine = cuisine_performance.index[0]
        
        insights.append(f"🍽️ **Best Performing Cuisine:** {best_cuisine} ({cuisine_performance.iloc[0]:.1f} stars)")
        
        # Display insights
        for i, insight in enumerate(insights):
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("🎯 Actionable Recommendations")
        
        recommendations = []
        
        if avg_rating < 4.0:
            recommendations.append("**Focus on Service Quality:** Low ratings often correlate with service issues.")
        
        if positive_pct < 60:
            recommendations.append("**Improve Customer Experience:** Address common complaints in negative reviews.")
        
        # Find most complained about aspects
        negative_reviews = self.df[self.df['sentiment'] == 'Negative']['review_text']
        if not negative_reviews.empty:
            common_complaints = []
            complaint_keywords = {
                'service': ['service', 'staff', 'waiter', 'waitress', 'slow'],
                'food': ['food', 'taste', 'cold', 'bland', 'overcooked'],
                'cleanliness': ['dirty', 'clean', 'hygiene', 'messy'],
                'price': ['expensive', 'overpriced', 'costly', 'price']
            }
            
            for aspect, keywords in complaint_keywords.items():
                count = negative_reviews.str.lower().str.contains('|'.join(keywords)).sum()
                if count > 0:
                    common_complaints.append((aspect, count))
            
            if common_complaints:
                most_complained = max(common_complaints, key=lambda x: x[1])
                recommendations.append(f"**Priority Fix:** Address {most_complained[0]} issues (mentioned in {most_complained[1]} negative reviews)")
        
        recommendations.append("**Monitor Trends:** Set up alerts for sudden drops in sentiment scores.")
        recommendations.append("**Leverage Positive Feedback:** Use positive reviews for marketing and staff training.")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Export functionality
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = self.df.to_csv(index=False)
            st.download_button(
                label="📊 Download Full Analysis (CSV)",
                data=csv,
                file_name='restaurant_sentiment_analysis.csv',
                mime='text/csv'
            )
        
        with col2:
            # Create summary report
            summary_report = {
                'Total Reviews': len(self.df),
                'Average Rating': f"{avg_rating:.2f}",
                'Positive Sentiment %': f"{positive_pct:.1f}%",
                'Top Restaurant': best_restaurant,
                'Best Cuisine': best_cuisine
            }
            
            summary_df = pd.DataFrame(list(summary_report.items()), columns=['Metric', 'Value'])
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📋 Download Summary Report (CSV)",
                data=summary_csv,
                file_name='restaurant_analysis_summary.csv',
                mime='text/csv'
            )
    
    def run_dashboard(self):
        """Main function to run the entire dashboard"""
        # Sidebar for data loading
        if self.load_data():
            self.process_data()
            
            # Main dashboard tabs
            tabs = st.tabs(["📊 Overview", "💭 Sentiment Analysis", "🏪 Restaurant Analysis", "☁️ Word Analysis", "💡 Insights"])
            
            with tabs[0]:
                self.show_overview()
            
            with tabs[1]:
                self.show_sentiment_analysis()
            
            with tabs[2]:
                self.show_restaurant_analysis()
            
            with tabs[3]:
                self.show_word_analysis()
            
            with tabs[4]:
                self.show_insights()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #888;'>
            🍽️ Restaurant Review Sentiment Analyzer Dashboard<br>
            Built with Streamlit & Python • Data-driven restaurant insights
        </div>
        """, unsafe_allow_html=True)

# Main execution
def main():
    """Run the Streamlit dashboard"""
    dashboard = RestaurantDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()