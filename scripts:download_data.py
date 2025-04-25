"""
Data processing script for Yelp reviews dataset.
Processes raw Yelp reviews and business data into training format.
"""

import os
import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from tqdm import tqdm

# Template categories for question generation
QUESTION_TEMPLATES = {
    'general': [
        # General restaurant recommendations
        "What are some highly rated restaurants in {city}?",
        "Can you recommend restaurants with good reviews in {city}?",
        "Where can I find well-reviewed places to eat in {city}?",
        "Looking for restaurants with positive feedback in {city}, any suggestions?",
        "What restaurants do you recommend in {city}?",
        "Got any food spots in {city}?",
        "Where do locals usually eat in {city}?",
        "I'm starving — what's good in {city}?",
        "Any hidden gems in {city} I should know about?",
        "What's your go-to restaurant in {city}?"
    ],
    'rating': [
        "What are some {rating}-star restaurants you'd recommend in {city}?",
        "Can you suggest restaurants rated {rating} stars in {city}?",
        "Where can I find {rating}-star dining options in {city}?",
        "Looking for {rating}-star restaurants in {city}, any recommendations?",
        "Which restaurants in {city} have {rating}-star ratings?",
        "Where can I eat without risking less than a {rating}-star experience?",
        "Only looking for {rating}-star vibes — got any in {city}?",
        "{rating}-stars or bust — what are my options in {city}?"
    ],
    'cuisine': [
        "What are some good {cuisine} restaurants in {city}?",
        "Can you recommend any {cuisine} places in {city}?",
        "Where can I find the best {cuisine} food in {city}?",
        "Looking for {cuisine} restaurants in {city}, any suggestions?",
        "Which {cuisine} restaurants in {city} do you recommend?",
        "Craving {cuisine} — who does it best in {city}?",
        "Any authentic {cuisine} joints around {city}?",
        "Got a fix for {cuisine} cravings in {city}?"
    ],
    'location': [
        "What are some good restaurants near {address}?",
        "Can you recommend places to eat around {address}?",
        "Where can I find restaurants in the {city} area?",
        "Looking for restaurants in {city}, any suggestions?",
        "What restaurants are located near {address}?",
        "Anywhere decent to grab food near {address}?",
        "I'm near {address}, what's good around here?",
        "Need lunch options close to {address} — help me out?"
    ],
    'hours': [
        "What restaurants are open {time} in {city}?",
        "Where can I eat {time} in {city}?",
        "Looking for places open {time} in {city}, any suggestions?",
        "Can you recommend restaurants open {time} in {city}?",
        "Which restaurants in {city} are open {time}?",
        "I'm hungry now — who's open {time} in {city}?",
        "Late night bites in {city}? Who's open {time}?",
        "Early morning eats in {city}? What's open {time}?"
    ],
    'occasion': [
        "What are good restaurants for {occasion} in {city}?",
        "Where should I go for {occasion} in {city}?",
        "Can you recommend places for {occasion} in {city}?",
        "Looking for restaurants for {occasion} in {city}, any ideas?",
        "Which restaurants in {city} are good for {occasion}?",
        "Planning something for {occasion} — what works in {city}?",
        "Where should I go in {city} for a nice {occasion} dinner?",
        "Need a spot in {city} that fits {occasion} vibes."
    ],
    'price': [
        "What are some {price} restaurants in {city}?",
        "Can you recommend {price} places to eat in {city}?",
        "Where can I find {price} dining options in {city}?",
        "Looking for {price} restaurants in {city}, any suggestions?",
        "Which restaurants in {city} are {price}?",
        "Any good {price} spots that won't disappoint in {city}?",
        "Balling on a budget — where should I eat in {city}?",
        "Where can I splurge a bit in {city} for food?"
    ],
    'dietary': [
        "What restaurants in {city} offer {dietary} options?",
        "Can you recommend {dietary} restaurants in {city}?",
        "Where can I find {dietary} food in {city}?",
        "Looking for {dietary} restaurants in {city}, any suggestions?",
        "Which restaurants in {city} cater to {dietary} diets?",
        "Need {dietary}-friendly spots in {city} — help!",
        "Where do people go for {dietary} eats in {city}?",
        "Got any {dietary} food recommendations around {city}?"
    ],
    'ambiance': [
        "What restaurants in {city} have {ambiance} atmosphere?",
        "Can you recommend {ambiance} restaurants in {city}?",
        "Where can I find {ambiance} dining spots in {city}?",
        "Looking for {ambiance} restaurants in {city}, any suggestions?",
        "Which restaurants in {city} offer {ambiance} ambiance?",
        "Want something with a {ambiance} vibe in {city} — where to?",
        "Need a place in {city} that feels {ambiance}",
        "Trying to set the mood — got any {ambiance} places in {city}?"
    ]
}

def load_business_data(business_file: str) -> Dict[str, Dict]:
    """Load business data and create a mapping of business_id to business info."""
    business_map = {}
    with open(business_file, 'r') as f:
        for line in f:
            business = json.loads(line)
            business_map[business['business_id']] = business
    return business_map

def get_business_hours(hours: Dict[str, str], day: str) -> str:
    """Get business hours for a specific day."""
    if day in hours:
        return hours[day]
    return "regular hours"

def create_diverse_questions(reviews: List[Dict], business: Dict) -> Dict:
    """Generate diverse questions based on multiple reviews for a business."""
    questions = []
    
    # Aggregate review information
    review_texts = [r['text'] for r in reviews]
    avg_rating = sum(r['stars'] for r in reviews) / len(reviews)
    
    # Extract popular dishes and keywords
    keywords = {
        'dishes': set(),
        'atmosphere': set(),
        'service': set(),
        'price': set()
    }
    
    for text in review_texts:
        # Add keyword extraction logic here
        if 'atmosphere' in text.lower():
            keywords['atmosphere'].update(text.split())
        if '$' in text:
            keywords['price'].update(['affordable'] if text.count('$') <= 2 else ['upscale'])
    
    # Format price level
    price_level = '$' * (1 if 'affordable' in keywords['price'] else 3)
    
    # Create response template
    response_template = f"{business['name']} ({price_level}) at {business['address']} is a {business['categories'][0] if business['categories'] else 'local'} restaurant "
    response_template += f"rated {avg_rating:.1f}/5 stars based on {len(reviews)} reviews. "
    
    if keywords['atmosphere']:
        response_template += f"The atmosphere is {', '.join(list(keywords['atmosphere'])[:3])}. "
    
    # Add a summary of reviews
    response_template += "Highlights from reviews: " + "; ".join(review_texts[:3])
    
    # Generate questions for different aspects
    for template in QUESTION_TEMPLATES['general']:
        questions.append({
            "input": f"User: {template.format(city=business['city'])}",
            "output": f"Assistant: Here are some recommendations: {response_template}"
        })
    
    if 'categories' in business and business['categories']:
        for cuisine in business['categories']:
            template = random.choice(QUESTION_TEMPLATES['cuisine'])
            questions.append({
                "input": f"User: {template.format(cuisine=cuisine, city=business['city'])}",
                "output": f"Assistant: For {cuisine} food in {business['city']}, I recommend: {response_template}"
            })
    
    return questions

def process_yelp_data(input_file, business_file, max_reviews=None):
    """Process Yelp reviews and create conversation pairs."""
    print("Starting to process Yelp dataset...")
    
    # Load business data
    print("Loading business data...")
    business_map = load_business_data(business_file)
    
    # Group reviews by business
    print("Grouping reviews by business...")
    business_reviews = {}
    reviews_processed = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            if max_reviews and reviews_processed >= max_reviews:
                break
                
            review = json.loads(line)
            business_id = review['business_id']
            
            if business_id not in business_map:
                continue
                
            if business_id not in business_reviews:
                business_reviews[business_id] = []
            
            # Only include reviews with substantial content
            if len(review['text']) > 50 and review['stars'] >= 3:
                business_reviews[business_id].append(review)
                reviews_processed += 1
    
    # Generate conversations using aggregated reviews
    conversations = []
    for business_id, reviews in business_reviews.items():
        if len(reviews) >= 3:  # Only use businesses with at least 3 reviews
            business = business_map[business_id]
            qas = create_diverse_questions(reviews, business)
            conversations.extend(qas)
    
    # Convert to DataFrame and split into train/validation
    df = pd.DataFrame(conversations)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    return train_df, val_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process Yelp reviews dataset')
    parser.add_argument('--input_file', required=True, help='Path to Yelp reviews dataset file')
    parser.add_argument('--business_file', required=True, help='Path to Yelp business dataset file')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed data')
    parser.add_argument('--max_reviews', type=int, default=10000, help='Maximum number of reviews to process')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the data
    train_df, val_df = process_yelp_data(args.input_file, args.business_file, args.max_reviews)
    
    print("Saving processed data...")
    
    # Save training data
    print("Saving training data...")
    train_path = os.path.join(args.output_dir, 'train.json')
    with open(train_path, 'w') as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    
    # Save validation data
    print("Saving validation data...")
    val_path = os.path.join(args.output_dir, 'val.json')
    with open(val_path, 'w') as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    
    print("\nProcessing complete!")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"\nFiles saved in: {args.output_dir}") 