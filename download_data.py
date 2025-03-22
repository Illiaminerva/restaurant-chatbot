import os
import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from tqdm import tqdm

# Question templates for different aspects
QUESTION_TEMPLATES = {
    'general': [
        "What are some highly rated restaurants in {city}?",
        "Can you recommend restaurants with good reviews in {city}?",
        "Where can I find well-reviewed places to eat in {city}?",
        "Looking for restaurants with positive feedback in {city}, any suggestions?",
        "What restaurants do you recommend in {city}?"
    ],
    'rating': [
        "What are some {rating}-star restaurants you'd recommend in {city}?",
        "Can you suggest restaurants rated {rating} stars in {city}?",
        "Where can I find {rating}-star dining options in {city}?",
        "Looking for {rating}-star restaurants in {city}, any recommendations?",
        "Which restaurants in {city} have {rating}-star ratings?"
    ],
    'cuisine': [
        "What are some good {cuisine} restaurants in {city}?",
        "Can you recommend any {cuisine} places in {city}?",
        "Where can I find the best {cuisine} food in {city}?",
        "Looking for {cuisine} restaurants in {city}, any suggestions?",
        "Which {cuisine} restaurants in {city} do you recommend?"
    ],
    'location': [
        "What are some good restaurants near {address}?",
        "Can you recommend places to eat around {address}?",
        "Where can I find restaurants in the {city} area?",
        "Looking for restaurants in {city}, any suggestions?",
        "What restaurants are located near {address}?"
    ],
    'hours': [
        "What restaurants are open {time} in {city}?",
        "Where can I eat {time} in {city}?",
        "Looking for places open {time} in {city}, any suggestions?",
        "Can you recommend restaurants open {time} in {city}?",
        "Which restaurants in {city} are open {time}?"
    ],
    'occasion': [
        "What are good restaurants for {occasion} in {city}?",
        "Where should I go for {occasion} in {city}?",
        "Can you recommend places for {occasion} in {city}?",
        "Looking for restaurants for {occasion} in {city}, any ideas?",
        "Which restaurants in {city} are good for {occasion}?"
    ],
    'price': [
        "What are some {price} restaurants in {city}?",
        "Can you recommend {price} places to eat in {city}?",
        "Where can I find {price} dining options in {city}?",
        "Looking for {price} restaurants in {city}, any suggestions?",
        "Which restaurants in {city} are {price}?"
    ],
    'dietary': [
        "What restaurants in {city} offer {dietary} options?",
        "Can you recommend {dietary} restaurants in {city}?",
        "Where can I find {dietary} food in {city}?",
        "Looking for {dietary} restaurants in {city}, any suggestions?",
        "Which restaurants in {city} cater to {dietary} diets?"
    ],
    'ambiance': [
        "What restaurants in {city} have {ambiance} atmosphere?",
        "Can you recommend {ambiance} restaurants in {city}?",
        "Where can I find {ambiance} dining spots in {city}?",
        "Looking for {ambiance} restaurants in {city}, any suggestions?",
        "Which restaurants in {city} offer {ambiance} ambiance?"
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

def create_diverse_questions(review: Dict, business: Dict) -> Dict:
    """Generate diverse questions based on the review and business content."""
    questions = []
    
    # Generate general questions
    template = random.choice(QUESTION_TEMPLATES['general'])
    questions.append({
        "input": f"User: {template.format(city=business['city'])}",
        "output": f"Assistant: Based on the reviews, {business['name']} at {business['address']} is a great option. {review['text']}"
    })
    
    # Generate rating-based questions
    rating = business['stars']
    template = random.choice(QUESTION_TEMPLATES['rating'])
    questions.append({
        "input": f"User: {template.format(rating=rating, city=business['city'])}",
        "output": f"Assistant: Based on the reviews, {business['name']} is a {rating}-star restaurant at {business['address']}. {review['text']}"
    })
    
    # Generate cuisine-based questions if categories exist
    if 'categories' in business and business['categories']:
        cuisine = random.choice(business['categories'])
        template = random.choice(QUESTION_TEMPLATES['cuisine'])
        questions.append({
            "input": f"User: {template.format(cuisine=cuisine, city=business['city'])}",
            "output": f"Assistant: Based on the reviews, {business['name']} is a great {cuisine} restaurant at {business['address']}. {review['text']}"
        })
    
    # Generate location-based questions
    template = random.choice(QUESTION_TEMPLATES['location'])
    questions.append({
        "input": f"User: {template.format(address=business['address'], city=business['city'])}",
        "output": f"Assistant: Based on the reviews, {business['name']} at {business['address']} is a great option. {review['text']}"
    })
    
    # Generate hours-based questions if hours exist
    if 'hours' in business and business['hours']:
        day = random.choice(list(business['hours'].keys()))
        hours = get_business_hours(business['hours'], day)
        template = random.choice(QUESTION_TEMPLATES['hours'])
        questions.append({
            "input": f"User: {template.format(time=f'on {day} {hours}', city=business['city'])}",
            "output": f"Assistant: Based on the reviews, {business['name']} is open {hours} on {day} and is located at {business['address']}. {review['text']}"
        })
    
    # Generate occasion-based questions
    occasions = ["date night", "family dinner", "business lunch", "special celebration", "casual dining", "romantic dinner", "group gathering"]
    template = random.choice(QUESTION_TEMPLATES['occasion'])
    occasion = random.choice(occasions)
    questions.append({
        "input": f"User: {template.format(occasion=occasion, city=business['city'])}",
        "output": f"Assistant: Based on the reviews, {business['name']} at {business['address']} would be great for {occasion}. {review['text']}"
    })
    
    # Generate price-based questions
    price_levels = ["affordable", "moderate", "upscale", "luxury", "budget-friendly", "mid-range"]
    template = random.choice(QUESTION_TEMPLATES['price'])
    price = random.choice(price_levels)
    questions.append({
        "input": f"User: {template.format(price=price, city=business['city'])}",
        "output": f"Assistant: Based on the reviews, {business['name']} at {business['address']} is a {price} option. {review['text']}"
    })
    
    # Generate dietary-based questions
    dietary_options = ["vegetarian", "vegan", "gluten-free", "halal", "kosher", "organic", "farm-to-table"]
    template = random.choice(QUESTION_TEMPLATES['dietary'])
    dietary = random.choice(dietary_options)
    questions.append({
        "input": f"User: {template.format(dietary=dietary, city=business['city'])}",
        "output": f"Assistant: Based on the reviews, {business['name']} at {business['address']} offers {dietary} options. {review['text']}"
    })
    
    # Generate ambiance-based questions
    ambiance_types = ["casual", "formal", "romantic", "family-friendly", "outdoor seating", "cozy", "modern", "traditional"]
    template = random.choice(QUESTION_TEMPLATES['ambiance'])
    ambiance = random.choice(ambiance_types)
    questions.append({
        "input": f"User: {template.format(ambiance=ambiance, city=business['city'])}",
        "output": f"Assistant: Based on the reviews, {business['name']} at {business['address']} has a {ambiance} atmosphere. {review['text']}"
    })
    
    # Return a random question from the generated ones
    return random.choice(questions)

def process_yelp_data(input_file, business_file, max_reviews=None):
    """Process Yelp reviews and create conversation pairs."""
    print("Starting to process Yelp dataset...")
    
    # Load business data
    print("Loading business data...")
    business_map = load_business_data(business_file)
    
    conversations = []
    reviews_processed = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            if max_reviews and reviews_processed >= max_reviews:
                break
                
            review = json.loads(line)
            
            # Skip if we don't have business data
            if review['business_id'] not in business_map:
                continue
                
            business = business_map[review['business_id']]
            
            # Basic quality filter
            if len(review['text']) > 50 and review['stars'] >= 3:
                # Generate diverse questions for this review
                conversation = create_diverse_questions(review, business)
                conversations.append(conversation)
                reviews_processed += 1
                
                if reviews_processed % 1000 == 0:
                    print(f"Processed {reviews_processed} reviews...")
    
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