from flask import Flask, render_template, request, send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import re
import torch
from scipy.special import softmax
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use a non-GUI backend
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Load the saved model and tokenizer from the saved_bert_model directory
tokenizer = AutoTokenizer.from_pretrained('./saved_bert_model')
model = AutoModelForSequenceClassification.from_pretrained('./saved_bert_model')

# Define sentiment score function with truncation
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt', truncation=True, max_length=512)
    result = model(tokens)
    probabilities = softmax(result.logits.detach().numpy(), axis=1)
    sentiment = int(torch.argmax(result.logits)) + 1  # Scale from 1 to 5
    return sentiment

# Function to scrape reviews from the Yelp link
def get_reviews(yelp_url):
    r = requests.get(yelp_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class': regex})
    reviews = [result.text for result in results]
    return reviews

# Calculate average sentiment score and review count distribution
def analyze_reviews(reviews):
    sentiments = [sentiment_score(review) for review in reviews]
    average_score = sum(sentiments) / len(sentiments) if sentiments else 0
    rating_counts = Counter(sentiments)
    return average_score, rating_counts

# Generate a bar chart for review count distribution
def generate_distribution_chart(rating_counts):
    labels, values = zip(*sorted(rating_counts.items()))
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Star Rating')
    plt.ylabel('Number of Reviews')
    plt.title('Review Count Distribution')
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        yelp_url = request.form['yelp_url']
        reviews = get_reviews(yelp_url)

        # Analyze reviews
        average_score, rating_counts = analyze_reviews(reviews)
        
        # Prepare data for display
        results = {
            'total_reviews': len(reviews),
            'average_score': average_score,
            'rating_counts': rating_counts
        }

        return render_template('result.html', results=results)
    
    return render_template('index.html')

@app.route('/chart')
def chart():
    # Generate distribution chart from rating_counts passed in the request args
    rating_counts = request.args.get('rating_counts')
    if rating_counts:
        img = generate_distribution_chart(rating_counts)
        return send_file(img, mimetype='image/png')
    else:
        return "No data available for chart", 400

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)