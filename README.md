# Yelp Recommendation Chatbot

A GPT-2 based chatbot that provides restaurant recommendations based on Yelp reviews.

## Project Structure

```
restaurant-bot/
├── src/
│   ├── models/
│   │   └── chatbot.py         # Main chatbot model implementation
│   └── training/
│       └── train_baseline.py  # Training script
├── scripts/
│   └── download_data.py      # Script to process Yelp data
├── data/                     # Place Yelp dataset files here
├── chat_with_bot.py          # Interactive chat interface
└── train_in_colab.ipynb      # Google Colab training notebook
```

## Training in Google Colab

1. Open `train_in_colab.ipynb` in Google Colab

2. Upload Project Files:
   ```
   src/
   ├── models/
   │   └── chatbot.py
   └── training/
       └── train_baseline.py
   download_data.py
   ```
   
3. Download and Upload Yelp Data:
   - Download the [Yelp Open Dataset](https://www.yelp.com/dataset)
   - You need both:
     - `yelp_academic_dataset_review.json`
     - `yelp_academic_dataset_business.json`
   - Create a `data` directory in Colab
   - Upload both files to the `data` directory

4. Process the Data:
   ```python
   !python download_data.py \
     --input_file data/yelp_academic_dataset_review.json \
     --business_file data/yelp_academic_dataset_business.json \
     --output_dir data \
     --max_reviews 10000
   ```
   This creates:
   - `data/train.json` (8000 examples)
   - `data/val.json` (1000 examples)

5. Run Training:
   - Follow the notebook cells in order
   - The model will be saved as `models/baseline/best_model.pt`
   - Download this file after training completes

## Model Features
- Restaurant-specific recommendations
- Sentiment analysis for responses
- Emoji feedback based on sentiment (to be improved)

## Initial Results and Analysis

### Training Performance
Our initial training shows promising results with consistent improvement across 8 epochs:
   - Training loss improved from 2.86 to 1.95
   - Validation loss improved from 2.33 to 2.14
   - Model consistently saved better checkpoints throughout training

### Current Challenges
   - Location mismatches (e.g., recommending Santa Barbara restaurants for San Francisco queries)
   - Sometimes includes irrelevant review details
   - Responses contain redundant information

## Next Steps and Improvements

### 1. Reinforcement Learning Integration
Planning to implement:
- Location accuracy rewards
- Response relevance scoring
- Conciseness rewards
- User satisfaction metrics

### 2. Response Quality Enhancement
- Better filtering of review content
- Location-based scoring system
- Improved context understanding
- More focused response generation
