# Import statements
from flask import Flask, render_template, request, send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import torch
import requests
import json
import time
import random
from collections import Counter

# Initialize Flask application
app = Flask(__name__)

# Load pre-trained BERT model and tokenizer for sentiment analysis
# Model should be stored locally in ./bert_model directory
tokenizer = AutoTokenizer.from_pretrained('./bert_model')
model = AutoModelForSequenceClassification.from_pretrained('./bert_model')

# Use GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Map numerical labels to sentiment categories
label_to_sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}

class YelpScraper:
    """
    A class to scrape business IDs & reviews from Yelp search results pages.
    Uses Selenium WebDriver to handle JavaScript-rendered content.
    """
    def __init__(self):
        # Configure Chrome WebDriver with anti-detection measures
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Run browser in headless mode
        options.add_argument('--headless')
        
        self.driver = webdriver.Chrome(options=options)

        # Set maximum wait time for elements
        self.wait = WebDriverWait(self.driver, 20)

    def random_sleep(self, min_seconds=2, max_seconds=5):
        """Add random delays between actions to mimic human behavior"""
        time.sleep(random.uniform(min_seconds, max_seconds))

    def get_place_ids(self, url):
        """
        Extract business IDs from a Yelp search results page
        
        Args:
            url (str): Yelp search results URL
            
        Returns:
            list: List of unique business IDs
        """
        try:
            self.driver.get(url)
            place_ids = []
            
            # Handle cookie consent popup if present
            try:
                cookie_button = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-cookie-banner-accept]"))
                )
                cookie_button.click()
            except:
                pass
            
            # Wait for business listings to load
            self.wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, "div[class*='businessName'], div[class*='container'] a[href*='/biz/']"
            )))
            
            self.random_sleep()
            
            # Scroll page to load more results
            for _ in range(5):
                current_height = self.driver.execute_script("return window.pageYOffset;")
                window_height = self.driver.execute_script("return window.innerHeight;")
                scroll_height = current_height + window_height
                self.driver.execute_script(f"window.scrollTo(0, {scroll_height});")
                self.random_sleep(1, 3)
            
            # Find all business links
            business_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                "a[href*='/biz/'], div[class*='container'] a[href*='/biz/']"
            )
            
            # Extract unique business IDs from URLs
            for element in business_elements:
                try:
                    href = element.get_attribute('href')
                    if href and '/biz/' in href:
                        place_id = href.split('/biz/')[1].split('?')[0].split('#')[0]
                        if place_id and place_id not in place_ids:
                            place_ids.append(place_id)
                except Exception:
                    continue

            return place_ids

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return []

    def close(self):
        """Clean up WebDriver resources"""
        self.driver.quit()

def scrape_reviews(place_id):
    """
    Fetch reviews for a business using the  Yelp Reviews API
    
    Args:
        place_id (str): Yelp business ID
        
    Returns:
        list: List of review texts
    """
    api_key = '1a9538cb0fc40b912f9c0b07756e120670d162b56e5985d81d244c919918e7b4'
    num = 49
    start = 0
    all_comments = []

    while True:
        params = {
            "api_key": api_key,
            "engine": "yelp_reviews",
            "place_id": place_id,
            "start": start,
            "num": num,
        }
        
        search = requests.get("https://serpapi.com/search", params=params)
        response = search.json()

        # Extract review texts
        for review in response['reviews']:
            if 'comment' in review and 'text' in review['comment']:
                all_comments.append(review['comment']['text'])

        # Break if we've received fewer reviews than requested (end of results)
        if len(response['reviews']) < num:
            break
            
        start += num
    
    return all_comments

def analyze_reviews(reviews):
    """
    Perform sentiment analysis on a list of reviews using BERT model
    
    Args:
        reviews (list): List of review texts
        
    Returns:
        dict: Contains sentiment distribution, confidence scores, and predictions
    """
    predictions = []
    confidence_scores = []
    
    for review in reviews:
        # Tokenize the review for BERT model
        inputs = tokenizer(
            review,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get predictions from model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            # Get the predicted label and confidence score
            predicted_label = torch.argmax(probs, dim=1).item()
            confidence_score = probs[0, predicted_label].item()
            
            predictions.append(predicted_label)
            confidence_scores.append(confidence_score)
    
    # Calculate sentiment distribution
    sentiment_counts = Counter(predictions)
    sentiment_distribution = {
        label_to_sentiment[label]: count 
        for label, count in sentiment_counts.items()
    }
    
    # Calculate average confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    return {
        'sentiment_distribution': sentiment_distribution,
        'average_confidence': avg_confidence,
        'predictions': [label_to_sentiment[p] for p in predictions],
        'confidence_scores': confidence_scores
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handle main page requests and form submissions
    GET: Display input form
    POST: Process Yelp URL and display analysis results
    """
    if request.method == 'POST':
        yelp_url = request.form['yelp_url']
        
        # Initialize scraper and get place IDs
        scraper = YelpScraper()
        try:
            place_ids = scraper.get_place_ids(yelp_url)
            
            if place_ids:
                # Get reviews for the first place ID
                reviews = scrape_reviews(place_ids[0])
                
                # Analyze sentiments of reviews
                results = analyze_reviews(reviews)
                
                # Save data to file for potential later use
                output_data = {
                    "reviews": reviews,
                    "analysis": results
                }
                with open('yelp_reviews.json', 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                # Prepare results for display
                display_results = {
                    'total_reviews': len(reviews),
                    'sentiment_distribution': results['sentiment_distribution'],
                    'average_confidence': round(results['average_confidence'], 2)
                }
                
                return render_template('result.html', results=display_results)
            else:
                return "No business found at the provided URL", 400
                
        finally:
            scraper.close()
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)