import os
import json
import kaggle
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import re
import time

# Create data directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Define available datasets with their details
DATASETS = {
    '1': {
        'name': 'UK Online Retail',
        'kaggle_id': 'mathchi/online-retail-ii-data-set',
        'description': 'E-commerce data with customer transactions from UK retailer',
        'processing_func': 'process_uk_retail',
        'output_file': 'ecommerce_data.csv',
        'size': '~22MB'
    },
    '2': {
        'name': 'Amazon Product Reviews',
        'kaggle_id': 'cynthiarempel/amazon-us-customer-reviews-dataset',
        'description': 'Customer reviews across multiple product categories',
        'processing_func': 'process_amazon_reviews',
        'output_file': 'amazon_reviews.csv',
        'size': '~50MB for sample'
    },
    '3': {
        'name': 'MovieLens Dataset',
        'kaggle_id': 'grouplens/movielens-20m-dataset',
        'description': 'Movie ratings dataset with 20 million ratings',
        'processing_func': 'process_movielens',
        'output_file': 'movie_ratings.csv',
        'size': '~190MB'
    },
    '4': {
        'name': 'Instacart Market Basket Analysis',
        'kaggle_id': 'instacart/instacart-market-basket-analysis',
        'description': 'Grocery shopping behavior with 3 million orders',
        'processing_func': 'process_instacart',
        'output_file': 'instacart_orders.csv',
        'size': '~500MB'
    },
    '5': {
        'name': 'H&M Personalized Fashion Recommendations',
        'kaggle_id': 'retailrocket/ecommerce-dataset',
        'description': 'Retail Rocket e-commerce dataset with behavioral data',
        'processing_func': 'process_retail_rocket',
        'output_file': 'retail_rocket_data.csv',
        'size': '~100MB'
    }
}

# Processing functions for each dataset
def process_uk_retail(input_file_path):
    print("\nProcessing UK Retail Dataset...")
    
    # Read and process the dataset (first 15000 records)
    df = pd.read_excel(input_file_path, nrows=15000)
    
    # Basic data cleaning
    df = df.dropna()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate purchase amount
    df['PurchaseAmount'] = df['Quantity'] * df['Price']
    
    # Remove cancelled orders (those with 'C' in InvoiceNo)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Remove negative quantities and prices
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    # Add a simple category based on the Description
    def assign_category(desc):
        desc = str(desc).lower()
        if any(word in desc for word in ['box', 'storage', 'holder']):
            return 'Storage'
        elif any(word in desc for word in ['garden', 'flower', 'plant']):
            return 'Garden'
        elif any(word in desc for word in ['christmas', 'valentine', 'easter']):
            return 'Seasonal'
        elif any(word in desc for word in ['bag', 'handbag', 'pouch']):
            return 'Bags'
        else:
            return 'Home & Living'
    
    df['Category'] = df['Description'].apply(assign_category)
    
    # Add a rating column (synthetic, based on price and quantity)
    df['Rating'] = ((df['Price'] / df['Price'].max() * 2.5) + 
                   (df['Quantity'] / df['Quantity'].max() * 2.5)).clip(1, 5)
    
    # Select and rename columns to match our dashboard's expected format
    processed_df = df[['CustomerID', 'Description', 'Category', 'Price', 
                       'Rating', 'InvoiceDate', 'PurchaseAmount']].copy()
    processed_df.columns = ['user_id', 'product_name', 'category', 'price',
                           'rating', 'purchase_date', 'purchase_amount']
    
    # Add search query (synthetic, based on product description)
    processed_df['search_query'] = processed_df['product_name'].apply(
        lambda x: ' '.join(str(x).lower().split()[:3]))
    
    return processed_df

def process_amazon_reviews(file_path):
    print("\nProcessing Amazon Reviews Dataset...")
    
    # Choose a single category file (e.g., 'Apparel')
    # The dataset has multiple category files, we'll use one for simplicity
    df = pd.read_csv(file_path, nrows=10000)  # Limit rows to keep file size manageable
    
    # Basic cleaning
    df = df.dropna(subset=['review_body', 'product_title', 'star_rating'])
    
    # Convert date
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    # Rename columns to match expected format
    renamed_df = df[['customer_id', 'product_title', 'product_category', 'product_parent',
                     'star_rating', 'review_date', 'helpful_votes']].copy()
    
    renamed_df.columns = ['user_id', 'product_name', 'category', 'product_id',
                         'rating', 'purchase_date', 'helpful_votes']
    
    # Add synthetic purchase amount based on rating
    renamed_df['purchase_amount'] = renamed_df['rating'] * np.random.uniform(5, 50, len(renamed_df))
    
    # Add search query based on product name
    renamed_df['search_query'] = renamed_df['product_name'].apply(
        lambda x: ' '.join(str(x).lower().split()[:3]))
    
    return renamed_df

def process_movielens(ratings_file, movies_file):
    print("\nProcessing MovieLens Dataset...")
    
    # Read ratings and movies data
    ratings = pd.read_csv(ratings_file, nrows=100000)  # Limit rows for manageable size
    movies = pd.read_csv(movies_file)
    
    # Merge datasets
    df = pd.merge(ratings, movies, on='movieId')
    
    # Extract year from title
    df['year'] = df['title'].str.extract('\((\d{4})\)', expand=False)
    df['title'] = df['title'].str.replace('\s*\(\d{4}\)', '', regex=True)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Extract genres
    df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if '|' in x else x)
    
    # Rename columns
    renamed_df = df[['userId', 'title', 'main_genre', 'movieId', 'rating', 'timestamp']].copy()
    renamed_df.columns = ['user_id', 'product_name', 'category', 'product_id', 
                         'rating', 'purchase_date']
    
    # Add synthetic purchase amount
    renamed_df['purchase_amount'] = renamed_df['rating'] * np.random.uniform(5, 15, len(renamed_df))
    
    # Add search query
    renamed_df['search_query'] = renamed_df['product_name'].apply(
        lambda x: ' '.join(str(x).lower().split()[:3]))
    
    return renamed_df

def process_instacart(orders_file, products_file):
    print("\nProcessing Instacart Dataset...")
    
    # Read orders and products
    orders = pd.read_csv(orders_file, nrows=10000)
    products = pd.read_csv(products_file)
    
    # Merge datasets
    df = pd.merge(orders, products, on='product_id')
    
    # Basic cleaning
    df = df.dropna(subset=['product_name'])
    
    # Convert dates
    df['order_time'] = pd.to_datetime(df['order_hour_of_day'], unit='h')
    
    # Rename columns
    renamed_df = df[['user_id', 'product_name', 'department', 'product_id', 
                    'order_time']].copy()
    renamed_df.columns = ['user_id', 'product_name', 'category', 'product_id', 
                         'purchase_date']
    
    # Add synthetic rating and purchase amount
    renamed_df['rating'] = np.random.uniform(3, 5, len(renamed_df))
    renamed_df['purchase_amount'] = np.random.uniform(2, 20, len(renamed_df))
    
    # Add search query
    renamed_df['search_query'] = renamed_df['product_name'].apply(
        lambda x: ' '.join(str(x).lower().split()[:3]))
    
    return renamed_df

def process_retail_rocket(file_path):
    print("\nProcessing Retail Rocket Dataset...")
    
    # Read events data
    df = pd.read_csv(file_path, nrows=20000)
    
    # Basic cleaning
    df = df.dropna()
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Create synthetic product names and categories
    product_names = {}
    categories = ['Electronics', 'Computers', 'Phones', 'Accessories', 'Home Appliances']
    
    for product_id in df['itemid'].unique():
        product_names[product_id] = f"Product {product_id} {np.random.choice(categories)}"
    
    df['product_name'] = df['itemid'].map(product_names)
    df['category'] = df['product_name'].apply(lambda x: x.split()[-1])
    
    # Rename columns
    renamed_df = df[['visitorid', 'product_name', 'category', 'itemid', 'timestamp']].copy()
    renamed_df.columns = ['user_id', 'product_name', 'category', 'product_id', 'purchase_date']
    
    # Add synthetic rating and purchase amount
    renamed_df['rating'] = np.random.uniform(1, 5, len(renamed_df))
    renamed_df['purchase_amount'] = np.random.uniform(10, 100, len(renamed_df))
    
    # Add search query
    renamed_df['search_query'] = renamed_df['product_name'].apply(
        lambda x: ' '.join(str(x).lower().split()[:3]))
    
    return renamed_df

def download_dataset(dataset_id):
    dataset = DATASETS.get(dataset_id)
    if not dataset:
        print(f"Dataset ID {dataset_id} not found!")
        return None
    
    print(f"\nDownloading {dataset['name']} from Kaggle...")
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset['kaggle_id'],
            path='data/raw',
            unzip=True
        )
        
        print(f"Dataset downloaded successfully!")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None

def process_dataset(dataset_id):
    dataset = DATASETS.get(dataset_id)
    if not dataset:
        return None
    
    processing_func = dataset['processing_func']
    output_file = f"data/raw/{dataset['output_file']}"
    
    try:
        # UK Retail Dataset processing
        if processing_func == 'process_uk_retail':
            input_file = 'data/raw/online_retail_II.xlsx'
            processed_df = process_uk_retail(input_file)
        
        # Amazon Reviews processing
        elif processing_func == 'process_amazon_reviews':
            # Find the downloaded CSV file (may vary depending on category)
            csv_files = [f for f in os.listdir('data/raw') if f.endswith('.csv') and 'amazon' in f.lower()]
            if not csv_files:
                print("Amazon reviews CSV file not found!")
                return None
            input_file = f"data/raw/{csv_files[0]}"
            processed_df = process_amazon_reviews(input_file)
        
        # MovieLens processing
        elif processing_func == 'process_movielens':
            ratings_file = 'data/raw/ratings.csv'
            movies_file = 'data/raw/movies.csv'
            processed_df = process_movielens(ratings_file, movies_file)
        
        # Instacart processing
        elif processing_func == 'process_instacart':
            orders_file = 'data/raw/order_products__prior.csv'
            products_file = 'data/raw/products.csv'
            processed_df = process_instacart(orders_file, products_file)
        
        # Retail Rocket processing
        elif processing_func == 'process_retail_rocket':
            events_file = 'data/raw/events.csv'
            processed_df = process_retail_rocket(events_file)
        
        else:
            print(f"Unknown processing function: {processing_func}")
            return None
        
        # Save the processed dataset
        processed_df.to_csv(output_file, index=False)
        
        print(f"\nDataset Statistics:")
        print("-" * 50)
        print(f"Total Records: {len(processed_df)}")
        print(f"Total Users: {processed_df['user_id'].nunique()}")
        print(f"Total Products: {processed_df['product_name'].nunique()}")
        print(f"Categories: {', '.join(processed_df['category'].unique()[:5])}" + 
              (" ..." if len(processed_df['category'].unique()) > 5 else ""))
        
        if 'purchase_date' in processed_df.columns:
            print(f"Date Range: {processed_df['purchase_date'].min().date()} to {processed_df['purchase_date'].max().date()}")
        
        if 'price' in processed_df.columns:
            print(f"Price Range: ${processed_df['price'].min():.2f} - ${processed_df['price'].max():.2f}")
        elif 'purchase_amount' in processed_df.columns:
            print(f"Purchase Amount Range: ${processed_df['purchase_amount'].min():.2f} - ${processed_df['purchase_amount'].max():.2f}")
        
        if 'rating' in processed_df.columns:
            print(f"Average Rating: {processed_df['rating'].mean():.2f}")
        
        print(f"\nProcessed dataset saved to: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None

def display_dataset_menu():
    print("\n" + "=" * 80)
    print("Available Kaggle Datasets for Product Recommendation Systems")
    print("=" * 80)
    
    for id, dataset in DATASETS.items():
        print(f"{id}. {dataset['name']} ({dataset['size']})")
        print(f"   {dataset['description']}")
        print(f"   Kaggle ID: {dataset['kaggle_id']}")
        print()
    
    print("=" * 80)

def main():
    print("\n=== Kaggle Dataset Downloader for Product Recommendation Systems ===")
    print("\nBefore running this script, make sure you have:")
    print("1. Created a Kaggle account")
    print("2. Downloaded your API token (kaggle.json)")
    print("3. Placed kaggle.json in ~/.kaggle/")
    print("4. Made the API token file read-only: chmod 600 ~/.kaggle/kaggle.json")
    
    input("\nPress Enter to continue when you're ready...")
    
    try:
        display_dataset_menu()
        
        # Get user input for dataset selection
        while True:
            selected_ids = input("\nEnter the ID(s) of the dataset(s) you want to download (comma-separated, e.g., 1,3,5): ")
            selected_ids = [id.strip() for id in selected_ids.split(',')]
            
            if all(id in DATASETS for id in selected_ids):
                break
            else:
                print("Invalid dataset ID(s). Please try again.")
        
        # Download and process selected datasets
        for dataset_id in selected_ids:
            print(f"\nProcessing dataset {dataset_id}: {DATASETS[dataset_id]['name']}")
            downloaded = download_dataset(dataset_id)
            
            if downloaded:
                time.sleep(1)  # Small delay to ensure files are completely downloaded
                output_file = process_dataset(dataset_id)
                if output_file:
                    print(f"Successfully processed {DATASETS[dataset_id]['name']}!")
                else:
                    print(f"Failed to process {DATASETS[dataset_id]['name']}.")
            else:
                print(f"Failed to download {DATASETS[dataset_id]['name']}.")
        
        print("\nAll selected datasets have been processed.")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease make sure you have:")
        print("- A valid Kaggle account")
        print("- Properly configured API token")
        print("- Internet connection")

if __name__ == "__main__":
    main()
