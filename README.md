
# Yelp Restaurant Review Sentiment Analysis

This project is a web-based sentiment analysis tool for restaurant reviews from Yelp. Using a fine-tuned BERT model, it scrapes reviews from Yelp, processes them, and provides an average sentiment score along with a review count distribution.

## Project Structure

```
MLProject/
├── bert_model/               # Directory for saved BERT model files
│   ├── bert_model_code.ipynb # Bert model fine tuning code
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── test_results.csv
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
└── templates/                # HTML templates for Flask
    ├── index.html
    └── result.html
├── app.py                    # Main Flask app with all routes and logic
├── README.md           
├── requirements.txt          # requirements.txt
├── yelp_reviews.json         # Output file containing all reviews.      
```

## Prerequisites

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/installation/) for managing Python packages

## Setup Instructions

### 1. Download the Pretrained BERT Model

Download the fine-tuned BERT model files from [MODEL_DOWNLOAD_WEBSITE](https://drive.google.com/file/d/1XfXyQnV5-dsCQZaEPl0wBLkW-ZwmHruk/view?usp=sharing) and place them in the `./bert_model/` directory in the project folder. **This step is required before running the application, or else it will give an error.**

### 2. Install Required Python Packages

Navigate to the project directory and install the required dependencies.

```bash
pip install -r requirements.txt
```

Alternatively, you can install packages manually:

```bash
pip install flask transformers requests beautifulsoup4 torch scipy matplotlib
```

### 3. Run the Application

Start the Flask application by running `app.py`.

```bash
python3 app.py
```

### 4. Access the Web Application

Once the server is running, open a web browser and navigate to `http://127.0.0.1:5000` to access the application. If you want to access it from other devices on your local network, find your local IP address (e.g., `192.168.x.x`) and use it as follows: `http://192.168.x.x:5000`.

## Usage Instructions

1. **Enter a Yelp Link**: On the main page, input a Yelp restaurant URL in the form and click "Analyze." Note: remove extra part for better performance. Yelp restaurant link format 'https://www.yelp.com/biz/business-name'. Eg. In this link "https://www.yelp.com/biz/the-whiskey-coop-syracuse?osq=Restaurants", remove "?osq=Restaurants". 
2. **View Results**: The application will scrape and analyze all reviews, then display:
   - **Total Reviews Analyzed**: Displays the total number of reviews analyzed from the dataset or input.
   - **Average Sentiment Score**: The average sentiment confidence score (from 0.0 to 1.0), calculated based on the sentiment of all reviews.
   - **Sentiment Distribution**: A breakdown of the number of reviews for each sentiment category (e.g., "Positive", "Negative", etc.) displayed as a list.

## Files and Directories

- **`app.py`**: Main application file, contains the Flask server setup and routes for scraping reviews, processing sentiment, and displaying results.
- **`bert_model/`**: Directory containing the downloaded BERT model files.
- **`templates/`**: Contains the HTML files (`index.html` for the input form and `result.html` for displaying the output).

## Troubleshooting

If you encounter issues with Matplotlib or plotting, ensure that the non-GUI backend (`Agg`) is set up correctly in `app.py`.

If you see `RuntimeError` or other errors related to model size or tokenization, double-check the sequence truncation settings in the `sentiment_score` function in `app.py`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

With these instructions, you should be able to set up, run, and access the web-based sentiment analysis tool locally.
