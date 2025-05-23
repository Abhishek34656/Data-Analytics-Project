# Product Recommendation System

## Project Overview
This project implements a sophisticated product recommendation system using customer behavior data, purchase history, and search patterns. The system analyzes user interactions and preferences to provide personalized product recommendations through an interactive dashboard.

## Project Structure
```
product_recommendation_system/
├── dashboard/           # Dashboard implementation
│   ├── app.py           # Main dashboard application
│   ├── assets/          # Dashboard styling
│   └── take_screenshots.py  # Utility for dashboard screenshots
├── data/                # Data directory
│   ├── processed/       # Processed and human-readable datasets
│   │   ├── human_readable_data.csv  # Clean, human-readable data
│   │   └── processed_data.csv       # Original processed dataset
│   └── raw/             # Raw data and data acquisition
│       └── download_kaggle_dataset.py  # Kaggle data downloader
└── scripts/             # Utility scripts
    ├── convert_to_readable.py     # Script to convert data to human-readable format
    └── download_retail_dataset.py  # Script to download e-commerce dataset
```

## Features
- Data collection and preprocessing
- User behavior analysis
- Product similarity computation
- Personalized recommendation generation
- Interactive visualization dashboard
- Outlier detection and handling
- Feature engineering

## Dataset
- 12,000 e-commerce transactions
- 5 product categories (Electronics, Fashion, Home & Living, Books, Sports)
- 999 unique users
- 50 unique products
- 1 year of transaction history
- Includes: purchase amounts, ratings, search queries
- Features: User interactions, product details, search history, purchase patterns

## Requirements
```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0
plotly==5.13.0
dash==2.8.0
seaborn==0.12.0
matplotlib==3.7.0
```

## Setup and Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure your Kaggle API credentials are set up in `~/.kaggle/kaggle.json`

## Usage
1. **Data Acquisition**: 
   - For new data: `python scripts/download_retail_dataset.py`
   - For human-readable conversion: `python scripts/convert_to_readable.py`

2. **Launch Dashboard**:
   - Run: `python dashboard/app.py`
   - Open browser at: http://127.0.0.1:8050

## Dashboard

The product recommendation dashboard provides comprehensive analytics with these key features:

### Dashboard Visualizations

1. **Key Metrics Overview**
   - Total sales figures
   - Average product ratings
   - Unique product count
   - Top performing product categories

2. **Sales Distribution Analysis**
   - Purchase amount distribution with statistical insights
   - Mean purchase value indicator
   - Transaction frequency analysis

3. **Product Ratings Visualization**
   - Distribution of ratings across products (1-5 stars)
   - Average rating indicators
   - Rating trends by category

4. **Category Distribution**
   - Interactive donut chart showing product category breakdown
   - Percentage distribution across categories
   - Visual hierarchy of product types

5. **Top Selling Products**
   - Horizontal bar chart of best-performing products
   - Integrated star ratings display
   - Category color-coding for quick reference

6. **Monthly Sales Trends**
   - Time series analysis of sales performance
   - Order volume visualization
   - Dual-axis chart showing correlation between sales and order quantity

### Viewing Dashboard Visualizations

The dashboard visualizations are available in HTML format in the `images/html` directory. To view them:

1. Navigate to the `images/html` directory
2. Open any HTML file in a web browser
3. For a complete view, open `01_full_dashboard.html`
4. Individual component visualizations are available as separate HTML files

## Features
1. Interactive data visualization
2. Category-based filtering
3. Real-time chart updates
4. Purchase pattern analysis
5. Rating distribution insights
6. Product performance metrics

## Results
- Detailed visualizations in the dashboard
- Key metrics and insights
- Model performance analysis

## Contributing
Feel free to submit issues and enhancement requests.

## License
MIT License
