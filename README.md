
# Yelp Restaurant Review Sentiment Analysis

This project is a web-based sentiment analysis tool for restaurant reviews from Yelp. Using a fine-tuned BERT model, it scrapes reviews from Yelp, processes them, and provides an average sentiment score along with a review count distribution.

## Project Structure

```
ML_Project/
├── app.py                    # Main Flask app with all routes and logic
├── bert_model_only.py        # Script with model-related helper functions
├── Reviews.csv               # Dataset file for local testing (if needed)
├── saved_bert_model/         # Directory for saved BERT model files
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
└── templates/                # HTML templates for Flask
    ├── index.html
    └── result.html
```

## Prerequisites

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/installation/) for managing Python packages

## Setup Instructions

### 1. Download the Pretrained BERT Model

Download the fine-tuned BERT model files from [MODEL_DOWNLOAD_WEBSITE](https://model-download-placeholder.com) and place them in the `saved_bert_model/` directory in the project folder. **This step is required before running the application, or else it will give an error.**

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
python app.py
```

### 4. Access the Web Application

Once the server is running, open a web browser and navigate to `http://127.0.0.1:5000` to access the application. If you want to access it from other devices on your local network, find your local IP address (e.g., `192.168.x.x`) and use it as follows: `http://192.168.x.x:5000`.

## Usage Instructions

1. **Enter a Yelp Link**: On the main page, input a Yelp restaurant URL in the form and click "Analyze."
2. **View Results**: The application will scrape and analyze the reviews, then display:
   - **Average Sentiment Score**: The average rating from 1 (most negative) to 5 (most positive).
   - **Review Count Distribution**: The number of reviews that fall into each star rating.
   - A **graph** showing the review count distribution.

## Files and Directories

- **`app.py`**: Main application file, contains the Flask server setup and routes for scraping reviews, processing sentiment, and displaying results.
- **`bert_model_only.py`**: Helper script for model-related functions (if needed separately).
- **`saved_bert_model/`**: Directory containing the downloaded BERT model files.
- **`templates/`**: Contains the HTML files (`index.html` for the input form and `result.html` for displaying the output).

## Troubleshooting

If you encounter issues with Matplotlib or plotting, ensure that the non-GUI backend (`Agg`) is set up correctly in `app.py`.

If you see `RuntimeError` or other errors related to model size or tokenization, double-check the sequence truncation settings in the `sentiment_score` function in `app.py`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

With these instructions, you should be able to set up, run, and access the web-based sentiment analysis tool locally.
