#!/usr/bin/env python3
"""
Generate 10,000+ Restaurant Reviews CSV
Creates a large dataset with realistic reviews matching our dashboard format
Columns: review_text, rating, restaurant_name, cuisine_type
"""

import pandas as pd
import random
import itertools

# Comprehensive restaurant data
RESTAURANT_DATA = {
    'Italian': [
        "Mario's Italian Bistro", "Tony's Pizza Palace", "Bella Vista", "Nonna's Kitchen", 
        "Luigi's Trattoria", "La Dolce Vita", "Roma Restaurant", "Amore Mio", 
        "Il Giardino", "Pasta Paradise", "Giuseppe's", "Venetian Dreams", 
        "Tuscany Table", "Mama Rosa's", "Little Italy", "Olive Garden", 
        "Papa John's Pizza", "Domino's Pizza", "Pizza Hut", "Carrabba's"
    ],
    'Chinese': [
        "Golden Dragon", "Bamboo Garden", "China House", "Panda Express", 
        "Great Wall", "Jade Palace", "Dragon Phoenix", "Lucky Dragon", 
        "Imperial Garden", "China Town", "Red Lantern", "Golden Wok", 
        "Fortune Cookie", "Mandarin House", "Dynasty Restaurant", "Szechuan Palace",
        "P.F. Chang's", "Pei Wei", "Pick Up Stix", "Panda Garden"
    ],
    'American': [
        "Downtown Diner", "Corner Cafe", "Sunny Side Cafe", "Hillside Grill",
        "Local Burger Joint", "Garden Fresh", "Main Street Eatery", "City Diner",
        "Hometown Grill", "The Kitchen", "Comfort Food Co", "American Bistro",
        "McDonald's", "Burger King", "Wendy's", "Five Guys", "Shake Shack",
        "In-N-Out", "White Castle", "Hardee's"
    ],
    'Mexican': [
        "Casa Miguel", "El Sombrero", "La Cantina", "Salsa Verde", 
        "Rio Grande", "Taco Bell", "Chipotle", "Qdoba", "El Pollo Loco",
        "Del Taco", "Taco Cabana", "Moe's Southwest", "Baja Fresh",
        "Cafe Rio", "Rubio's", "El Torito", "Chevys", "On The Border"
    ],
    'Japanese': [
        "Sakura Sushi", "Tokyo Express", "Rising Sun", "Wasabi House",
        "Sushi Palace", "Zen Garden", "Kyoto Restaurant", "Osaka Grill",
        "Ninja Sushi", "Samurai Kitchen", "Mount Fuji", "Tokyo Bay",
        "Benihana", "Hibachi Grill", "Sapporo", "Kobe Steakhouse"
    ],
    'Thai': [
        "Thai Garden", "Bangkok Kitchen", "Spice Thai", "Golden Elephant",
        "Thai Palace", "Pad Thai House", "Bangkok Express", "Thai Orchid",
        "Emerald Thai", "Royal Thai", "Thai Basil", "Siam Square",
        "Thai Pepper", "Lotus Thai", "Thai Smile", "Bangkok Bistro"
    ],
    'Indian': [
        "Spice Route", "Curry Palace", "Taj Mahal", "Bombay Bistro",
        "Maharaja", "Bengal Tiger", "Saffron", "India House",
        "Karma Indian", "Tandoor Oven", "Curry Express", "Delhi Palace",
        "Bollywood Cafe", "Punjabi Kitchen", "Masala House", "India Gate"
    ],
    'French': [
        "Le Bernardin", "French Laundry", "Daniel", "Le Cirque",
        "Jean-Georges", "Cafe de Paris", "Bistro Le Petit", "La Brasserie",
        "Le Jardin", "Chez Pierre", "Maison Blanc", "Le Coq",
        "French Quarter", "Parisian Cafe", "Le Bistrot", "Cafe Monet"
    ],
    'Steakhouse': [
        "Prime Steakhouse", "Black Angus", "Ruth's Chris", "Morton's",
        "Capital Grille", "Outback Steakhouse", "Texas Roadhouse", "LongHorn",
        "Mastro's", "Del Frisco's", "Fleming's", "Ruth's Chris",
        "Chicago Cut", "Peter Luger", "Keen's", "Palm Restaurant"
    ],
    'Seafood': [
        "Ocean's Bounty", "Harbor View", "The Lobster Pot", "Captain's Table",
        "Fisherman's Wharf", "Red Lobster", "Joe's Crab Shack", "Bonefish Grill",
        "Legal Sea Foods", "McCormick & Schmick's", "Chart House", "Pier Market",
        "Neptune's", "Catch of the Day", "Anchor Bay", "Salty Pearl"
    ],
    'Korean': [
        "Seoul Kitchen", "Kimchi House", "Korean BBQ Palace", "Gangnam Style",
        "K-Town Grill", "Seoul Garden", "Arirang", "Han Yang",
        "Korea House", "Seoul Station", "Bulgogi Brothers", "Tofu House",
        "Hot Stone", "Korean Village", "Soju Bar", "Seoul Food"
    ],
    'Vietnamese': [
        "Pho Saigon", "Vietnam Kitchen", "Saigon Palace", "Pho 88",
        "Golden Pho", "Vietnam Garden", "Little Saigon", "Pho King",
        "Banh Mi Cafe", "Mekong River", "Hanoi Kitchen", "Pho Real",
        "Vietnam Express", "Saigon Star", "Pho Dynasty", "Vietnam Bistro"
    ],
    'Greek': [
        "Acropolis", "Zeus Taverna", "Greek Islands", "Santorini",
        "Parthenon", "Olive Tree", "Mykonos", "Athens Grill",
        "Greek Corner", "Aegean Sea", "Hellas Restaurant", "Crete Cafe",
        "Greek Village", "Sparta Grill", "Olympic Restaurant", "Delphi"
    ],
    'Brazilian': [
        "Rio Steakhouse", "Brazilian Grill", "Copacabana", "Samba Kitchen",
        "Ipanema", "Gaucho Grill", "Churrascaria", "Carnival",
        "Bossa Nova", "Tropical Grill", "Amazonia", "Brazil Nuts",
        "Carioca", "Favela Chic", "Brasil 66", "Rio Grande"
    ],
    'Mediterranean': [
        "Mediterranean Grill", "Cyprus Taverna", "Olive Branch", "Santorini Grill",
        "Athens Kitchen", "Mediterranean Coast", "Aegean Breeze", "Mykonos Cafe",
        "Crete Restaurant", "Rhodes Taverna", "Mediterranean Sea", "Olive Garden Med"
    ],
    'Middle Eastern': [
        "Pita Palace", "Hummus House", "Babylon Cafe", "Cedar Land",
        "Aladdin's", "Damascus", "Jerusalem", "Petra Kitchen",
        "Sultan's Palace", "Arabian Nights", "Oasis", "Desert Rose",
        "Marrakech", "Casablanca", "Istanbul", "Turkish Delight"
    ]
}

# Review templates with placeholders
POSITIVE_TEMPLATES = [
    "Absolutely {amazing_adj} experience! The {dish} was {food_adj} and the service was {service_adj}. {praise_detail} Will definitely be coming back!",
    "Outstanding {cuisine} cuisine! {dish} was cooked to perfection. The staff was {service_adj} and the atmosphere was {atmosphere_adj}. Highly recommend!",
    "Best {cuisine} food in the city! {food_praise} The {staff_member} was knowledgeable and made excellent recommendations. Perfect for {occasion}!",
    "Incredible dining experience! {food_praise} Service was {service_adj} and the restaurant has a {atmosphere_adj} atmosphere. Worth every penny!",
    "Love this place! {food_praise} {service_praise} Great value for the quality. Perfect spot for {occasion}.",
    "Fantastic restaurant! {dish} was {food_adj} and the {service_aspect} was {service_adj}. {praise_detail}",
    "Excellent {cuisine} restaurant! {food_praise} The ambiance is {atmosphere_adj} and perfect for {occasion}. Will be returning soon!",
    "Hidden gem! {dish} reminds me of {authentic_ref}. {service_praise} Reasonable prices for such quality.",
    "Outstanding meal! {food_praise} {service_praise} The {restaurant_feature} is impressive. Highly recommended!",
    "Perfect dining experience! {food_praise} Service was {service_adj} without being intrusive. Great for {occasion}.",
    "{amazing_adj} restaurant with authentic flavors! {dish} was prepared {cooking_method} and tasted {food_adj}. {praise_detail}",
    "Wow! This place exceeded all expectations. {food_praise} {service_praise} The {atmosphere_adj} atmosphere made it perfect for our {occasion}.",
    "Five stars all the way! {dish} was the best I've had in years. {service_praise} Will definitely become a regular customer!",
    "Incredible {cuisine} food! {food_praise} The chef really knows what they're doing. {praise_detail} Highly recommend to anyone!",
    "Amazing experience from start to finish! {service_praise} {dish} was {food_adj} and the {restaurant_feature} was impressive."
]

NEUTRAL_TEMPLATES = [
    "Decent {cuisine} food but nothing special. {dish} was okay but {minor_issue}. Service was adequate. {neutral_conclusion}",
    "Good restaurant with {positive_aspect} but {negative_aspect}. {dish} was {neutral_food_adj}. {neutral_service}",
    "Solid choice for {cuisine} food. {food_comment} Service is {neutral_service_adj}. {pricing_comment} Good option for {occasion}.",
    "Average experience overall. {positive_aspect} but {negative_aspect}. {dish} was {neutral_food_adj}. {neutral_conclusion}",
    "Nothing outstanding but nothing terrible either. {food_comment} {service_comment} {pricing_comment}",
    "Okay place for {cuisine} food. {dish} was {neutral_food_adj}. {service_comment} {atmosphere_comment}",
    "Decent option in the area. {positive_aspect} though {negative_aspect}. {pricing_comment} Would consider returning.",
    "Mixed experience. {positive_aspect} but {negative_aspect}. {dish} was {neutral_food_adj}. {neutral_service}",
    "Fair restaurant. {food_comment} {service_comment} Not bad but there are better options nearby.",
    "Standard {cuisine} fare. {dish} was {neutral_food_adj}. {service_comment} {pricing_comment}",
    "It's an okay place for {cuisine} food. {positive_aspect} but {negative_aspect}. {neutral_conclusion}",
    "Average dining experience. {dish} was {neutral_food_adj} and service was {neutral_service_adj}. {pricing_comment}",
    "Decent restaurant with room for improvement. {food_comment} {service_comment} Might give it another try.",
    "Nothing to write home about. {positive_aspect} but {negative_aspect}. {neutral_conclusion}",
    "Standard {cuisine} restaurant. {dish} was fine but {minor_issue}. Service was adequate."
]

NEGATIVE_TEMPLATES = [
    "Disappointing experience. {negative_food} and the service was {negative_service}. {complaint} Will not be returning.",
    "Terrible dining experience! {negative_food} {service_complaint} {atmosphere_complaint} Complete waste of money.",
    "Poor quality food and service. {dish} was {negative_food_adj} and {service_complaint}. {complaint}",
    "Very disappointed. {negative_food} Service was {negative_service}. {complaint} Overpriced for what you get.",
    "Awful restaurant. {negative_food} {service_complaint} {cleanliness_complaint} Would not recommend to anyone.",
    "Unacceptable experience. {wait_complaint} {negative_food} {service_complaint} Management needs to address these issues.",
    "Worst {cuisine} food I've had. {dish} was {negative_food_adj}. {service_complaint} {atmosphere_complaint}",
    "Complete disappointment. {negative_food} {service_complaint} For the price, expected much better quality.",
    "Poor experience from start to finish. {wait_complaint} {negative_food} {service_complaint} Avoid this place.",
    "Subpar restaurant. {negative_food} {service_complaint} {pricing_complaint} Many better options in the area.",
    "Horrible experience! {dish} was {negative_food_adj} and {service_complaint}. {complaint} Never going back!",
    "Absolutely terrible! {negative_food} {wait_complaint} The {staff_member} was {negative_service}. Save your money!",
    "Worst dining experience in years. {negative_food} {service_complaint} {complaint} How is this place still open?",
    "Complete disaster! {dish} was {negative_food_adj} and arrived {timing_issue}. {service_complaint}",
    "Terrible {cuisine} restaurant. {negative_food} {service_complaint} {atmosphere_complaint} Waste of time and money."
]

# Replacement words and phrases
REPLACEMENTS = {
    'amazing_adj': ['incredible', 'outstanding', 'phenomenal', 'exceptional', 'remarkable', 'superb', 'fantastic', 'wonderful', 'magnificent', 'spectacular'],
    'food_adj': ['delicious', 'amazing', 'perfect', 'outstanding', 'incredible', 'exceptional', 'wonderful', 'superb', 'excellent', 'fantastic', 'flavorful', 'fresh', 'tasty', 'divine', 'exquisite'],
    'neutral_food_adj': ['okay', 'decent', 'fine', 'adequate', 'reasonable', 'satisfactory', 'fair', 'standard', 'typical', 'average', 'passable', 'acceptable'],
    'negative_food_adj': ['terrible', 'awful', 'horrible', 'disgusting', 'bland', 'tasteless', 'overcooked', 'cold', 'stale', 'disappointing', 'gross', 'inedible', 'salty', 'bitter', 'soggy'],
    'service_adj': ['excellent', 'outstanding', 'professional', 'friendly', 'attentive', 'knowledgeable', 'efficient', 'courteous', 'helpful', 'welcoming', 'responsive', 'gracious'],
    'neutral_service_adj': ['adequate', 'decent', 'okay', 'fair', 'standard', 'typical', 'average', 'reasonable'],
    'negative_service': ['slow', 'rude', 'unprofessional', 'inattentive', 'dismissive', 'terrible', 'poor', 'awful', 'incompetent', 'unfriendly', 'negligent', 'disrespectful'],
    'atmosphere_adj': ['romantic', 'cozy', 'elegant', 'welcoming', 'vibrant', 'relaxing', 'sophisticated', 'charming', 'intimate', 'lively', 'warm', 'inviting', 'comfortable', 'stylish'],
    'dish': ['pasta', 'steak', 'salmon', 'chicken', 'pizza', 'soup', 'salad', 'burger', 'sushi', 'curry', 'tacos', 'sandwich', 'seafood', 'risotto', 'lasagna', 'pad thai', 'ramen', 'dim sum'],
    'staff_member': ['server', 'waiter', 'waitress', 'host', 'chef', 'manager', 'sommelier', 'bartender'],
    'occasion': ['date night', 'family dinner', 'business lunch', 'anniversary', 'birthday celebration', 'casual dining', 'special occasion', 'lunch meeting', 'dinner with friends', 'romantic evening'],
    'food_praise': ['The flavors were incredible', 'Every dish was perfectly seasoned', 'Fresh ingredients throughout', 'Authentic preparation', 'Creative and delicious dishes', 'Outstanding quality ingredients', 'Perfectly cooked', 'Amazing presentation'],
    'service_praise': ['Staff was attentive and friendly', 'Our server was knowledgeable', 'Excellent customer service', 'Professional and efficient staff', 'Warm and welcoming service', 'Quick and courteous service'],
    'praise_detail': ['The wine pairing was perfect', 'Great attention to detail', 'Fresh ingredients throughout', 'Beautifully presented dishes', 'Generous portion sizes', 'Reasonable prices', 'Clean and comfortable environment'],
    'service_aspect': ['service', 'wait staff', 'management', 'kitchen', 'hostess'],
    'restaurant_feature': ['wine selection', 'dessert menu', 'cocktail list', 'atmosphere', 'decor', 'location', 'music', 'lighting'],
    'authentic_ref': ['my grandmother\'s cooking', 'food from Italy', 'street food in Bangkok', 'home cooking', 'my trip to Mexico', 'traditional recipes'],
    'cooking_method': ['perfectly', 'expertly', 'beautifully', 'skillfully', 'traditionally', 'authentically'],
    'negative_food': ['Food was cold when it arrived', 'Dishes lacked flavor', 'Poor quality ingredients', 'Food was overpriced', 'Preparation was sloppy', 'Everything tasted the same', 'Food was greasy'],
    'service_complaint': ['service was extremely slow', 'staff was rude and unhelpful', 'waited too long for our order', 'server was inattentive', 'poor customer service', 'staff seemed overwhelmed', 'no one checked on us'],
    'wait_complaint': ['Waited over an hour for food', 'Long wait despite reservation', 'Extremely slow service', 'Unacceptable wait times', 'Food took forever to arrive'],
    'complaint': ['Manager was unprofessional', 'Restaurant was dirty', 'Music was too loud', 'Tables were not cleaned properly', 'Overpriced for the quality', 'Kitchen was clearly understaffed', 'Poor value for money'],
    'atmosphere_complaint': ['Restaurant was too noisy', 'Uncomfortable seating', 'Poor lighting', 'Place needs renovation', 'Too crowded', 'Uncomfortable temperature'],
    'cleanliness_complaint': ['Restaurant was not clean', 'Tables were sticky', 'Bathrooms were dirty', 'Kitchen appeared unsanitary', 'Floors were filthy'],
    'timing_issue': ['20 minutes late', 'cold', 'at different times', 'burnt', 'undercooked'],
    'minor_issue': ['a bit salty', 'slightly overcooked', 'could use more seasoning', 'was a bit dry', 'lacked flavor'],
    'positive_aspect': ['good portion sizes', 'nice atmosphere', 'friendly staff', 'convenient location', 'reasonable prices', 'good selection'],
    'negative_aspect': ['prices are high', 'service was slow', 'limited parking', 'can get crowded', 'loud environment', 'small portions'],
    'food_comment': ['Food was tasty but nothing extraordinary', 'Dishes were well-prepared', 'Menu has good variety', 'Quality was consistent'],
    'service_comment': ['Service was consistent', 'Staff was polite but not exceptional', 'Servers were adequate', 'No complaints about service'],
    'pricing_comment': ['Prices are reasonable', 'Good value for money', 'A bit expensive but fair', 'Pricing is competitive', 'Worth the cost'],
    'neutral_service': ['Service was fine', 'Staff was adequate', 'No complaints about service', 'Servers did their job'],
    'neutral_conclusion': ['Worth trying once', 'Might return', 'Decent option in the area', 'Could be better', 'Nothing special but okay'],
    'atmosphere_comment': ['Nice ambiance', 'Comfortable setting', 'Pleasant atmosphere', 'Good decor'],
    'pricing_complaint': ['Way too expensive', 'Overpriced for the quality', 'Not worth the cost', 'Poor value for money']
}

def fill_template(template, cuisine, restaurant_name):
    """Fill a review template with random replacements"""
    review = template.format(cuisine=cuisine.lower())
    
    # Replace all placeholders
    for key, values in REPLACEMENTS.items():
        placeholder = '{' + key + '}'
        while placeholder in review:
            review = review.replace(placeholder, random.choice(values), 1)
    
    return review

def generate_reviews(num_reviews=10000):
    """Generate realistic restaurant reviews"""
    reviews = []
    restaurants_cycle = []
    
    # Create a balanced list of restaurants
    for cuisine, restaurant_list in RESTAURANT_DATA.items():
        for restaurant in restaurant_list:
            restaurants_cycle.extend([(restaurant, cuisine)] * (num_reviews // (len(RESTAURANT_DATA) * len(restaurant_list)) + 1))
    
    random.shuffle(restaurants_cycle)
    
    print(f"🚀 Generating {num_reviews:,} restaurant reviews...")
    
    for i in range(num_reviews):
        restaurant_name, cuisine = restaurants_cycle[i % len(restaurants_cycle)]
        
        # Realistic sentiment distribution
        rand = random.random()
        if rand < 0.55:  # 55% positive
            rating = random.choices([4, 5], weights=[0.4, 0.6])[0]
            template = random.choice(POSITIVE_TEMPLATES)
        elif rand < 0.80:  # 25% neutral
            rating = 3
            template = random.choice(NEUTRAL_TEMPLATES)
        else:  # 20% negative
            rating = random.choices([1, 2], weights=[0.4, 0.6])[0]
            template = random.choice(NEGATIVE_TEMPLATES)
        
        # Fill template
        review_text = fill_template(template, cuisine, restaurant_name)
        
        # Add occasional extra details
        if random.random() < 0.15:  # 15% chance
            extras = [
                " Great portion sizes!", " Parking was easy to find.", 
                " Perfect location.", " They take reservations.", 
                " Good for groups.", " Kid-friendly environment.",
                " Great happy hour deals.", " Live music on weekends.",
                " Outdoor seating available.", " BYOB policy.",
                " Dietary restrictions accommodated well.", " Fast WiFi available."
            ]
            review_text += random.choice(extras)
        
        reviews.append({
            'review_text': review_text,
            'rating': rating,
            'restaurant_name': restaurant_name,
            'cuisine_type': cuisine
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"   ✅ Generated {i + 1:,} reviews...")
    
    return reviews

def save_large_dataset():
    """Generate and save large restaurant dataset"""
    print("🍽️ Large Restaurant Reviews Dataset Generator")
    print("=" * 50)
    
    # Generate reviews
    reviews = generate_reviews(10000)
    
    # Create DataFrame
    df = pd.DataFrame(reviews)
    
    # Add some variety to ratings based on restaurant quality
    restaurant_quality = {}
    for restaurant in df['restaurant_name'].unique():
        # Some restaurants are consistently better
        base_quality = random.choice([1.0, 1.0, 1.0, 1.1, 1.2, 0.9, 0.8])  # Most are average
        restaurant_quality[restaurant] = base_quality
    
    # Adjust some ratings based on restaurant quality
    for idx, row in df.iterrows():
        quality_modifier = restaurant_quality[row['restaurant_name']]
        if quality_modifier > 1.0 and row['rating'] >= 3:
            if random.random() < 0.3:  # 30% chance to boost
                df.at[idx, 'rating'] = min(5, row['rating'] + 1)
        elif quality_modifier < 1.0 and row['rating'] <= 3:
            if random.random() < 0.3:  # 30% chance to lower
                df.at[idx, 'rating'] = max(1, row['rating'] - 1)
    
    # Save to CSV
    filename = 'restaurant_reviews_10000.csv'
    df.to_csv(filename, index=False)
    
    print(f"\n🎉 SUCCESS! Generated large dataset")
    print(f"📁 Saved as: {filename}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Total Reviews: {len(df):,}")
    print(f"   • Unique Restaurants: {df['restaurant_name'].nunique()}")
    print(f"   • Cuisine Types: {df['cuisine_type'].nunique()}")
    print(f"   • Average Rating: {df['rating'].mean():.2f}")
    print(f"   • File Size: ~{len(df) * 200 // 1024:.1f} KB")
    
    print(f"\n⭐ Rating Distribution:")
    for rating in sorted(df['rating'].unique()):
        count = (df['rating'] == rating).sum()
        percentage = (count / len(df)) * 100
        bar = '█' * int(percentage // 2)
        print(f"   {rating} stars: {count:4d} ({percentage:4.1f}%) {bar}")
    
    print(f"\n🍽️ Cuisine Distribution:")
    cuisine_counts = df['cuisine_type'].value_counts()
    for cuisine, count in cuisine_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"   {cuisine}: {count:4d} ({percentage:4.1f}%)")
    
    print(f"\n🎯 Ready to use!")
    print(f"   1. python3 run_analysis.py --dashboard")
    print(f"   2. Upload '{filename}'")
    print(f"   3. Analyze {len(df):,} professional reviews!")
    
    # Show sample
    print(f"\n📝 Sample Reviews:")
    for i in range(3):
        review = df.iloc[i]
        print(f"\n   {i+1}. {review['restaurant_name']} ({review['cuisine_type']}) - {review['rating']}⭐")
        print(f"      \"{review['review_text'][:100]}...\"")
    
    return filename

if __name__ == "__main__":
    save_large_dataset()