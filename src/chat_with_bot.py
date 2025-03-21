from src.models.chatbot import RestaurantChatbot
import torch

def chat_with_bot():
    # Initialize model with the trained weights
    model = RestaurantChatbot(device='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('models/baseline/best_model.pt'))
    model.eval()
    
    print("Restaurant Chatbot is ready! Type 'quit' to exit.")
    print("Ask me about restaurant recommendations, cuisine types, or specific dishes.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
            
        # Generate response
        response = model.generate_response(
            user_input,
            max_length=128,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        
        # Get sentiment score
        sentiment = model.get_sentiment(response)
        sentiment_str = "😊" if sentiment > 0.6 else "😐" if sentiment > 0.4 else "😟"
        
        print(f"\nBot: {response} {sentiment_str}")

if __name__ == "__main__":
    chat_with_bot() 