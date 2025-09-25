"""
Real Data Collection Module for Restaurant Reviews
Supports multiple data sources: Yelp API, Google Places, Web Scraping, and Datasets
"""

import requests
import pandas as pd
import time
import json
from bs4 import BeautifulSoup
import random
from urllib.parse import urlencode
import os

class RestaurantDataCollector:
    """Collect restaurant review data from multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    # Method 1: Yelp API (Requires API Key)
    def collect_yelp_reviews(self, api_key, location="New York, NY", term="restaurants", limit=50):
        """
        Collect reviews using Yelp API
        Get API key from: https://www.yelp.com/developers/v3/manage_app
        """
        print(f"🔍 Collecting Yelp data for {location}...")
        
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        # First, search for businesses
        search_url = "https://api.yelp.com/v3/businesses/search"
        search_params = {
            'location': location,
            'term': term,
            'limit': limit,
            'sort_by': 'review_count'
        }
        
        try:
            response = self.session.get(search_url, headers=headers, params=search_params)
            response.raise_for_status()
            businesses = response.json()['businesses']
            
            all_reviews = []
            
            for business in businesses[:20]:  # Limit to avoid rate limiting
                business_id = business['id']
                
                # Get reviews for each business
                reviews_url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
                reviews_response = self.session.get(reviews_url, headers=headers)
                
                if reviews_response.status_code == 200:
                    reviews_data = reviews_response.json()['reviews']
                    
                    for review in reviews_data:
                        all_reviews.append({
                            'review_text': review['text'],
                            'rating': review['rating'],
                            'restaurant_name': business['name'],
                            'cuisine_type': business.get('categories', [{}])[0].get('title', 'Unknown'),
                            'location': business['location']['city'],
                            'review_date': review['time_created']
                        })
                
                time.sleep(0.2)  # Rate limiting
            
            print(f"✅ Collected {len(all_reviews)} Yelp reviews")
            return pd.DataFrame(all_reviews)
            
        except Exception as e:
            print(f"❌ Error collecting Yelp data: {e}")
            return pd.DataFrame()
    
    # Method 2: Web Scraping TripAdvisor (Educational purposes)
    def scrape_tripadvisor_reviews(self, city="new-york", max_pages=3):
        """
        Scrape restaurant reviews from TripAdvisor
        Note: For educational purposes only. Respect robots.txt and terms of service
        """
        print(f"🔍 Scraping TripAdvisor reviews for {city}...")
        
        base_url = f"https://www.tripadvisor.com/Restaurants-g{self.get_city_code(city)}-{city}.html"
        all_reviews = []
        
        try:
            # This is a simplified example - in practice you'd need to handle
            # JavaScript rendering, pagination, and anti-bot measures
            response = self.session.get(base_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract restaurant links (this would need to be updated based on current HTML structure)
            restaurant_links = soup.find_all('a', class_='restaurant-link')[:10]
            
            for link in restaurant_links:
                restaurant_url = "https://www.tripadvisor.com" + link.get('href')
                
                # Get restaurant page
                restaurant_response = self.session.get(restaurant_url)
                restaurant_soup = BeautifulSoup(restaurant_response.content, 'html.parser')
                
                # Extract reviews (simplified - actual structure varies)
                reviews = restaurant_soup.find_all('div', class_='review-container')[:5]
                
                for review in reviews:
                    try:
                        review_text = review.find('p').get_text(strip=True) if review.find('p') else ""
                        rating_elem = review.find('span', class_='ui_bubble_rating')
                        rating = 3  # Default rating if not found
                        
                        if review_text:
                            all_reviews.append({
                                'review_text': review_text,
                                'rating': rating,
                                'restaurant_name': 'TripAdvisor Restaurant',
                                'cuisine_type': 'Various',
                                'location': city,
                                'source': 'TripAdvisor'
                            })
                    except:
                        continue
                
                time.sleep(1)  # Be respectful with requests
            
            print(f"✅ Collected {len(all_reviews)} TripAdvisor reviews")
            return pd.DataFrame(all_reviews)
            
        except Exception as e:
            print(f"❌ Error scraping TripAdvisor: {e}")
            return pd.DataFrame()
    
    def get_city_code(self, city):
        """Get TripAdvisor city code - simplified mapping"""
        city_codes = {
            'new-york': '60763',
            'los-angeles': '32655',
            'chicago': '35805',
            'miami': '34438'
        }
        return city_codes.get(city, '60763')
    
    # Method 3: Download Existing Datasets
    def download_kaggle_dataset(self, dataset_name="yelp_reviews.csv"):
        """
        Download restaurant review datasets from Kaggle or other sources
        Popular datasets:
        - Yelp Open Dataset
        - Amazon Fine Food Reviews
        - Google Local Reviews
        """
        print("📥 Downloading dataset...")
        
        # Sample URLs for restaurant review datasets
        dataset_urls = {
            "yelp_reviews.csv": "https://github.com/ankitjha31/Yelp-Dataset/raw/master/yelp_dataset.csv",
            "restaurant_reviews.csv": "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv"
        }
        
        try:
            if dataset_name in dataset_urls:
                url = dataset_urls[dataset_name]
                df = pd.read_csv(url)
                
                # Standardize column names
                if 'text' in df.columns:
                    df = df.rename(columns={'text': 'review_text'})
                if 'stars' in df.columns:
                    df = df.rename(columns={'stars': 'rating'})
                
                print(f"✅ Downloaded {len(df)} reviews from {dataset_name}")
                return df
            else:
                print(f"❌ Dataset {dataset_name} not found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return pd.DataFrame()
    
    # Method 4: Google Places API (Alternative to Yelp)
    def collect_google_places_reviews(self, api_key, location="New York, NY", radius=5000):
        """
        Collect reviews using Google Places API
        Get API key from: https://console.cloud.google.com/
        """
        print(f"🔍 Collecting Google Places data for {location}...")
        
        # Search for restaurants
        search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        search_params = {
            'query': f'restaurants in {location}',
            'key': api_key
        }
        
        try:
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            places = response.json()['results']
            
            all_reviews = []
            
            for place in places[:20]:  # Limit to avoid quota issues
                place_id = place['place_id']
                
                # Get detailed place information including reviews
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_params = {
                    'place_id': place_id,
                    'fields': 'name,reviews,types,rating',
                    'key': api_key
                }
                
                details_response = self.session.get(details_url, params=details_params)
                
                if details_response.status_code == 200:
                    details = details_response.json()['result']
                    
                    if 'reviews' in details:
                        for review in details['reviews']:
                            all_reviews.append({
                                'review_text': review['text'],
                                'rating': review['rating'],
                                'restaurant_name': details['name'],
                                'cuisine_type': details.get('types', ['restaurant'])[0],
                                'location': location,
                                'review_date': review['time']
                            })
                
                time.sleep(0.1)  # Rate limiting
            
            print(f"✅ Collected {len(all_reviews)} Google Places reviews")
            return pd.DataFrame(all_reviews)
            
        except Exception as e:
            print(f"❌ Error collecting Google Places data: {e}")
            return pd.DataFrame()
    
    # Method 5: Load Local Files
    def load_local_data(self, file_path):
        """Load data from local CSV, JSON, or Excel files"""
        print(f"📁 Loading data from {file_path}...")
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"✅ Loaded {len(df)} reviews from local file")
            return df
            
        except Exception as e:
            print(f"❌ Error loading local data: {e}")
            return pd.DataFrame()
    
    def collect_all_sources(self, config):
        """
        Collect data from all available sources based on configuration
        
        config = {
            'yelp_api_key': 'your_key',
            'google_api_key': 'your_key', 
            'location': 'New York, NY',
            'use_datasets': True,
            'use_scraping': False,  # Set to True if you want to try scraping
            'local_files': ['reviews.csv']
        }
        """
        all_dataframes = []
        
        # Yelp API
        if config.get('yelp_api_key'):
            yelp_df = self.collect_yelp_reviews(
                config['yelp_api_key'], 
                config.get('location', 'New York, NY')
            )
            if not yelp_df.empty:
                yelp_df['source'] = 'Yelp'
                all_dataframes.append(yelp_df)
        
        # Google Places API  
        if config.get('google_api_key'):
            google_df = self.collect_google_places_reviews(
                config['google_api_key'],
                config.get('location', 'New York, NY')
            )
            if not google_df.empty:
                google_df['source'] = 'Google'
                all_dataframes.append(google_df)
        
        # Download datasets
        if config.get('use_datasets'):
            dataset_df = self.download_kaggle_dataset()
            if not dataset_df.empty:
                dataset_df['source'] = 'Dataset'
                all_dataframes.append(dataset_df)
        
        # Web scraping (use cautiously)
        if config.get('use_scraping'):
            scraped_df = self.scrape_tripadvisor_reviews(
                config.get('location', 'new-york')
            )
            if not scraped_df.empty:
                scraped_df['source'] = 'TripAdvisor'
                all_dataframes.append(scraped_df)
        
        # Local files
        if config.get('local_files'):
            for file_path in config['local_files']:
                if os.path.exists(file_path):
                    local_df = self.load_local_data(file_path)
                    if not local_df.empty:
                        local_df['source'] = 'Local'
                        all_dataframes.append(local_df)
        
        # Combine all data
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Clean and standardize
            combined_df = self.clean_combined_data(combined_df)
            
            print(f"🎉 Total collected: {len(combined_df)} reviews from {len(all_dataframes)} sources")
            return combined_df
        else:
            print("❌ No data collected from any source")
            return pd.DataFrame()
    
    def clean_combined_data(self, df):
        """Clean and standardize the combined dataset"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['review_text'], keep='first')
        
        # Ensure required columns exist
        required_columns = ['review_text', 'rating', 'restaurant_name', 'cuisine_type']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 'Unknown'
        
        # Clean text
        df['review_text'] = df['review_text'].astype(str)
        df['review_text'] = df['review_text'].str.strip()
        
        # Filter out very short reviews
        df = df[df['review_text'].str.len() >= 10]
        
        # Ensure rating is numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        
        return df

# Example usage and configuration
def main():
    """Example of how to collect real restaurant data"""
    
    collector = RestaurantDataCollector()
    
    # Configuration - Update with your API keys
    config = {
        # Get Yelp API key from: https://www.yelp.com/developers/v3/manage_app
        'yelp_api_key': None,  # Replace with your Yelp API key
        
        # Get Google API key from: https://console.cloud.google.com/
        'google_api_key': None,  # Replace with your Google API key
        
        'location': 'New York, NY',
        'use_datasets': True,  # This will work without API keys
        'use_scraping': False,  # Set to True if you want to try scraping (be careful!)
        'local_files': []  # Add paths to any local CSV files you have
    }
    
    # Method 1: Try to collect from all sources
    print("🚀 Starting data collection from all available sources...")
    df = collector.collect_all_sources(config)
    
    # Method 2: If APIs aren't available, just use datasets
    if df.empty:
        print("⚠️  No API keys provided. Using dataset instead...")
        df = collector.download_kaggle_dataset()
    
    # Save the collected data
    if not df.empty:
        df.to_csv('collected_restaurant_reviews.csv', index=False)
        print(f"💾 Saved {len(df)} reviews to 'collected_restaurant_reviews.csv'")
        
        # Show sample
        print(f"\n📋 Sample data:")
        print(df.head())
        
        return df
    else:
        print("❌ No data was collected. Please check your configuration.")
        return None

if __name__ == "__main__":
    main()