#!/usr/bin/env python3
"""
Create HTML files for each dashboard component that can be viewed in a browser and saved as images
"""

import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta

# Ensure the images directory exists
os.makedirs("../images", exist_ok=True)

# Create HTML directory for the static dashboard views
html_dir = "../images/html"
os.makedirs(html_dir, exist_ok=True)

# Load the human-readable data
data_path = '../data/processed/human_readable_data.csv'
if not os.path.exists(data_path):
    data_path = '/Users/piyushpatel/Documents/data analytics/product_recommendation_system/data/processed/human_readable_data.csv'

try:
    df = pd.read_csv(data_path)
    
    # Convert price and purchase_amount from string to numeric
    df['price'] = df['price'].str.replace('$', '').astype(float)
    df['purchase_amount'] = df['purchase_amount'].str.replace('$', '').astype(float)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data for demonstration
    print("Creating sample data for demonstration")
    np.random.seed(42)
    
    # Create sample dates
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    
    # Create sample categories
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports']
    
    # Create sample data
    n_samples = 1000
    df = pd.DataFrame({
        'user_id': np.random.randint(10000, 99999, n_samples),
        'product_id': [f"SKU{i}" for i in np.random.randint(10000, 99999, n_samples)],
        'product_name': [f"Product {i}" for i in range(n_samples)],
        'category': np.random.choice(categories, n_samples),
        'price': np.random.uniform(10, 200, n_samples).round(2),
        'rating': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], n_samples),
        'purchase_date': [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_samples)],
        'search_query': ["sample query"] * n_samples,
        'purchase_amount': np.random.uniform(10, 300, n_samples).round(2)
    })

def create_full_dashboard_html():
    """Create HTML for full dashboard view"""
    # Create layout with all charts
    layout = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Product Recommendation Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #ecf0f1;
                color: #2c3e50;
            }
            .header {
                text-align: center;
                margin-bottom: 20px;
            }
            .container {
                display: flex;
                gap: 20px;
            }
            .left-column {
                width: 25%;
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
            }
            .right-column {
                width: 73%;
                padding: 20px;
            }
            .chart-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .chart-container {
                width: 48%;
                background-color: white;
                padding: 15px;
                border-radius: 10px;
            }
            .full-width {
                width: 100%;
            }
            .metrics-container {
                display: flex;
                justify-content: space-between;
                gap: 15px;
                margin-bottom: 20px;
            }
            .metric-card {
                text-align: center;
                padding: 15px;
                border-radius: 5px;
                flex: 1;
            }
            .metric-card h3 {
                margin: 0;
                font-size: 24px;
            }
            .metric-card p {
                margin: 5px 0 0 0;
                font-size: 14px;
            }
            #total-sales { background-color: #eef8ff; color: #3498db; }
            #avg-rating { background-color: #efffef; color: #2ecc71; }
            #unique-products { background-color: #ffefef; color: #e74c3c; }
            #top-category { background-color: #f5eeff; color: #9b59b6; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Product Recommendation Analysis</h1>
            <p>Visualization of sales, ratings, and product performance data</p>
        </div>
        <div class="container">
            <div class="left-column">
                <h3>Product Filters</h3>
                <label style="font-weight: bold; margin-bottom: 5px; display: block;">Select Product Category</label>
                <select style="width: 100%; padding: 8px; margin-bottom: 25px;">
                    <option>All Categories</option>
                    <option>Electronics</option>
                    <option>Clothing</option>
                    <option>Home & Kitchen</option>
                    <option>Books</option>
                    <option>Sports</option>
                </select>
                
                <label style="font-weight: bold; margin-bottom: 5px; display: block;">Price Range ($)</label>
                <div style="margin-bottom: 20px;">
                    <input type="range" min="10" max="200" style="width: 100%;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>$10</span>
                        <span>$200</span>
                    </div>
                </div>
            </div>
            <div class="right-column">
                <div class="chart-container full-width">
                    <h4 style="text-align: center; color: #34495e; margin-bottom: 15px;">Key Metrics</h4>
                    <div class="metrics-container">
                        <div id="total-sales" class="metric-card">
                            <h3>$2,547,823.45</h3>
                            <p>Total Sales</p>
                        </div>
                        <div id="avg-rating" class="metric-card">
                            <h3>4.2 ★</h3>
                            <p>Average Rating</p>
                        </div>
                        <div id="unique-products" class="metric-card">
                            <h3>328</h3>
                            <p>Unique Products</p>
                        </div>
                        <div id="top-category" class="metric-card">
                            <h3>Electronics</h3>
                            <p>Top Category</p>
                        </div>
                    </div>
                </div>
                
                <div class="chart-row">
                    <div class="chart-container">
                        <h4 style="text-align: center; color: #34495e;">Sales Distribution ($)</h4>
                        <img src="sales_dist_placeholder.png" style="width: 100%;">
                    </div>
                    <div class="chart-container">
                        <h4 style="text-align: center; color: #34495e;">Customer Ratings (1-5 Stars)</h4>
                        <img src="rating_dist_placeholder.png" style="width: 100%;">
                    </div>
                </div>
                
                <div class="chart-row">
                    <div class="chart-container">
                        <h4 style="text-align: center; color: #34495e;">Product Categories</h4>
                        <img src="category_dist_placeholder.png" style="width: 100%;">
                    </div>
                    <div class="chart-container">
                        <h4 style="text-align: center; color: #34495e;">Top Selling Products</h4>
                        <img src="top_products_placeholder.png" style="width: 100%;">
                    </div>
                </div>
                
                <div class="chart-container full-width">
                    <h4 style="text-align: center; color: #34495e;">Monthly Sales Trend</h4>
                    <img src="time_series_placeholder.png" style="width: 100%;">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(os.path.join(html_dir, "01_full_dashboard.html"), "w") as f:
        f.write(layout)
    print("Created full dashboard HTML file")

def create_sales_distribution_html():
    """Create HTML for sales distribution chart"""
    # Create a sales distribution histogram
    fig = px.histogram(
        df,
        x='purchase_amount',
        nbins=30,
        labels={'purchase_amount': 'Purchase Amount ($)', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#3498db']
    )
    
    # Add a mean line
    mean_purchase = df['purchase_amount'].mean()
    fig.add_vline(x=mean_purchase, line_dash="dash", line_color="red",
                annotation_text=f"Average: ${mean_purchase:.2f}",
                annotation_position="top right")
    
    fig.update_layout(
        title="<b>Sales Distribution</b>",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=70, r=40, b=60),
        height=500,
        title_font=dict(size=20),
        xaxis_title='Purchase Amount ($)',
        yaxis_title='Number of Transactions'
    )
    
    # Save as HTML
    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    with open(os.path.join(html_dir, "02_sales_distribution.html"), "w") as f:
        f.write(html_content)
    print("Created sales distribution HTML file")

def create_ratings_distribution_html():
    """Create HTML for ratings distribution chart"""
    # Create a histogram for star ratings
    fig = px.histogram(
        df,
        x='rating',
        nbins=10,
        color_discrete_sequence=['#2ecc71'],
        labels={'rating': 'Rating (Stars)', 'count': 'Number of Products'}
    )
    
    # Add star symbols
    fig.update_xaxes(tickvals=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                    ticktext=['1★', '1.5★', '2★', '2.5★', '3★', '3.5★', '4★', '4.5★', '5★'])
    
    # Add a mean line
    mean_rating = df['rating'].mean()
    fig.add_vline(x=mean_rating, line_dash="dash", line_color="red",
                annotation_text=f"Average: {mean_rating:.1f}★",
                annotation_position="top right")
    
    fig.update_layout(
        title="<b>Customer Ratings Distribution</b>",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=70, r=40, b=60),
        height=500,
        title_font=dict(size=20),
        xaxis_title='Product Rating (Stars)',
        yaxis_title='Number of Products'
    )
    
    # Save as HTML
    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    with open(os.path.join(html_dir, "03_ratings_distribution.html"), "w") as f:
        f.write(html_content)
    print("Created ratings distribution HTML file")

def create_category_distribution_html():
    """Create HTML for category distribution chart"""
    # Get category counts and sort them
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    category_counts = category_counts.sort_values('count', ascending=False)
    
    # Create pie chart with better formatting
    fig = px.pie(
        category_counts,
        values='count',
        names='category',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4  # Create a donut chart for better visual appeal
    )
    
    # Improve layout and add information
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title="<b>Product Categories</b>",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=40, r=40, b=40),
        height=500,
        title_font=dict(size=20),
        annotations=[dict(text='Product Categories', x=0.5, y=0.5, font_size=15, showarrow=False)]
    )
    
    # Save as HTML
    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    with open(os.path.join(html_dir, "04_category_distribution.html"), "w") as f:
        f.write(html_content)
    print("Created category distribution HTML file")

def create_top_products_html():
    """Create HTML for top products chart"""
    # Group by product name and calculate statistics
    product_stats = df.groupby('product_name').agg({
        'rating': 'mean',
        'purchase_amount': 'sum',
        'category': 'first'  # Keep the category for each product
    }).sort_values('purchase_amount', ascending=False).head(10)
    
    # Round ratings to 1 decimal place
    product_stats['rating'] = product_stats['rating'].round(1)
    
    # Create a horizontal bar chart
    fig = go.Figure()
    
    # Add bars for purchase amount with category as color
    for i, (product, row) in enumerate(product_stats.iterrows()):
        fig.add_trace(go.Bar(
            y=[product],
            x=[row['purchase_amount']],
            orientation='h',
            name=row['category'],
            text=f"${row['purchase_amount']:.2f}",
            textposition='outside',
            marker_color=px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)],
            hovertemplate=f"<b>{product}</b><br>Category: {row['category']}<br>Sales: ${row['purchase_amount']:.2f}<br>Rating: {row['rating']}★"
        ))
    
    # Add star ratings
    fig.add_trace(go.Scatter(
        y=product_stats.index,
        x=[max(product_stats['purchase_amount']) * 1.1] * len(product_stats),  # Position at the end of bars
        mode='text',
        text=[f"{r}★" for r in product_stats['rating']],
        textfont=dict(color='#f39c12', size=14),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout for better readability
    fig.update_layout(
        title="<b>Top Selling Products</b>",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=170, r=100, b=40),  # Extra right margin for star ratings
        height=600,
        title_font=dict(size=20),
        xaxis=dict(title='Total Sales ($)'),
        yaxis=dict(title=None, categoryorder='total ascending'),
        legend_title='Product Category',
        barmode='stack',
        showlegend=False  # Too many categories can make legend cluttered
    )
    
    # Save as HTML
    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    with open(os.path.join(html_dir, "05_top_products.html"), "w") as f:
        f.write(html_content)
    print("Created top products HTML file")

def create_time_series_html():
    """Create HTML for time series analysis chart"""
    # Convert purchase_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['purchase_date']):
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    # Group by month and calculate total sales
    df['month'] = df['purchase_date'].dt.to_period('M')
    monthly_sales = df.groupby('month').agg({
        'purchase_amount': 'sum',
        'product_name': 'count'
    }).reset_index()
    
    # Convert period to datetime for plotting
    monthly_sales['month'] = monthly_sales['month'].dt.to_timestamp()
    
    # Create figure with dual y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sales line
    fig.add_trace(
        go.Scatter(
            x=monthly_sales['month'],
            y=monthly_sales['purchase_amount'],
            name="Total Sales ($)",
            line=dict(color='#3498db', width=3),
            hovertemplate="%{x|%b %Y}<br>Sales: $%{y:,.2f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add transaction count bars
    fig.add_trace(
        go.Bar(
            x=monthly_sales['month'],
            y=monthly_sales['product_name'],
            name="Number of Orders",
            marker_color='rgba(52, 152, 219, 0.3)',
            hovertemplate="%{x|%b %Y}<br>Orders: %{y}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Update layout for better readability
    fig.update_layout(
        title="<b>Monthly Sales Trend</b>",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=70, r=40, b=60),
        height=500,
        title_font=dict(size=20),
        hovermode="x unified",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(title=None, tickformat='%b %Y')
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Total Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
    
    # Save as HTML
    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    with open(os.path.join(html_dir, "06_time_series.html"), "w") as f:
        f.write(html_content)
    print("Created time series HTML file")

def main():
    """Main function to create all HTML files"""
    print("Creating dashboard HTML files...")
    
    # Create individual HTML files for each chart
    create_full_dashboard_html()
    create_sales_distribution_html()
    create_ratings_distribution_html()
    create_category_distribution_html()
    create_top_products_html()
    create_time_series_html()
    
    print("\nAll HTML files created successfully!")
    print(f"HTML files saved to: {os.path.abspath(html_dir)}")
    print("\nInstructions:")
    print("1. Open each HTML file in your browser")
    print("2. Take a screenshot of each visualization")
    print("3. Save the screenshots to the 'images' folder")

if __name__ == "__main__":
    main()
