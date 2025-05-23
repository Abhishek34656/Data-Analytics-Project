#!/usr/bin/env python3
"""
Generate static images of the dashboard charts
"""

import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# Ensure the images directory exists
os.makedirs("../images", exist_ok=True)

# Load the human-readable data
data_path = '../data/processed/human_readable_data.csv'
if not os.path.exists(data_path):
    data_path = '/Users/piyushpatel/Documents/data analytics/product_recommendation_system/data/processed/human_readable_data.csv'

df = pd.read_csv(data_path)

# Convert price and purchase_amount from string to numeric
df['price'] = df['price'].str.replace('$', '').astype(float)
df['purchase_amount'] = df['purchase_amount'].str.replace('$', '').astype(float)
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

def generate_key_metrics_image():
    """Generate an image for key metrics"""
    # Calculate metrics
    total_sales = df['purchase_amount'].sum()
    avg_rating = df['rating'].mean()
    unique_products = df['product_name'].nunique()
    top_category = df['category'].value_counts().idxmax()
    
    # Create a figure
    fig = go.Figure()
    
    # Add a table with metrics
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            line_color='white',
            fill_color='#2c3e50',
            align='center',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[
                ["Total Sales", "Average Rating", "Unique Products", "Top Category"],
                [f"${total_sales:,.2f}", f"{avg_rating:.1f} ★", f"{unique_products:,}", top_category]
            ],
            line_color='white',
            fill_color=[['#eef8ff', '#efffef', '#ffefef', '#f5eeff']],
            align='center',
            font=dict(color='#333', size=14),
            height=40
        )
    ))
    
    fig.update_layout(
        title="<b>Product Recommendation Analysis - Key Metrics</b>",
        title_font=dict(size=20),
        width=900,
        height=250,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    
    # Save the figure
    fig.write_image("../images/01_key_metrics.png")
    print("Generated key metrics image")

def generate_sales_distribution():
    """Generate sales distribution histogram"""
    fig = px.histogram(
        df,
        x='purchase_amount',
        nbins=30,
        labels={'purchase_amount': 'Purchase Amount ($)', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#3498db'],
        title="<b>Sales Distribution</b>"
    )
    
    # Add a mean line
    mean_purchase = df['purchase_amount'].mean()
    fig.add_vline(x=mean_purchase, line_dash="dash", line_color="red",
                annotation_text=f"Average: ${mean_purchase:.2f}",
                annotation_position="top right")
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=70, r=40, b=60),
        width=900,
        height=500,
        title_font=dict(size=20),
        xaxis_title='Purchase Amount ($)',
        yaxis_title='Number of Transactions'
    )
    
    # Save the figure
    fig.write_image("../images/02_sales_distribution.png")
    print("Generated sales distribution image")

def generate_category_distribution():
    """Generate category distribution pie chart"""
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
        hole=0.4,  # Create a donut chart for better visual appeal
        title="<b>Product Categories</b>"
    )
    
    # Improve layout and add information
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=40, r=40, b=40),
        width=900,
        height=600,
        title_font=dict(size=20),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        annotations=[dict(text='Product Categories', x=0.5, y=0.5, font_size=15, showarrow=False)]
    )
    
    # Save the figure
    fig.write_image("../images/03_category_distribution.png")
    print("Generated category distribution image")

def generate_top_products():
    """Generate top products chart"""
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
        width=900,
        height=600,
        title_font=dict(size=20),
        xaxis=dict(title='Total Sales ($)'),
        yaxis=dict(title=None, categoryorder='total ascending'),
        legend_title='Product Category',
        barmode='stack',
        showlegend=False  # Too many categories can make legend cluttered
    )
    
    # Save the figure
    fig.write_image("../images/04_top_products.png")
    print("Generated top products image")

def generate_time_series():
    """Generate time series analysis"""
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
        width=900,
        height=500,
        title_font=dict(size=20),
        hovermode="x unified",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(title=None, tickformat='%b %Y')
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Total Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
    
    # Save the figure
    fig.write_image("../images/05_time_series.png")
    print("Generated time series image")

def generate_electronics_filtered():
    """Generate image with Electronics category filter"""
    # Filter for Electronics category
    electronics_df = df[df['category'] == 'Electronics']
    
    # Create sales distribution for Electronics
    fig = px.histogram(
        electronics_df,
        x='purchase_amount',
        nbins=30,
        labels={'purchase_amount': 'Purchase Amount ($)', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#e74c3c'],
        title="<b>Electronics Category - Sales Distribution</b>"
    )
    
    # Add a mean line
    mean_purchase = electronics_df['purchase_amount'].mean()
    fig.add_vline(x=mean_purchase, line_dash="dash", line_color="blue",
                annotation_text=f"Average: ${mean_purchase:.2f}",
                annotation_position="top right")
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=70, l=70, r=40, b=60),
        width=900,
        height=500,
        title_font=dict(size=20),
        xaxis_title='Purchase Amount ($)',
        yaxis_title='Number of Transactions'
    )
    
    # Save the figure
    fig.write_image("../images/06_electronics_filtered.png")
    print("Generated Electronics filtered image")

def main():
    """Main function to generate all images"""
    print("Generating dashboard images...")
    
    # Generate all images
    generate_key_metrics_image()
    generate_sales_distribution()
    generate_category_distribution()
    generate_top_products()
    generate_time_series()
    generate_electronics_filtered()
    
    print("\nAll images generated successfully!")
    print(f"Images saved to: {os.path.abspath('../images')}")

if __name__ == "__main__":
    main()
