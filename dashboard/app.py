import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Load the human-readable data
try:
    # Try the path in the product_recommendation_system directory first
    df = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\Projects\product_recommendation\Data-Analytics-Project\data\processed\human_readable_data.csv')
except FileNotFoundError:
    # Fall back to the file in the main data analytics directory
    df = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\Projects\product_recommendation\Data-Analytics-Project\data\processed\processed_data.csv')

# Convert price and purchase_amount from string to numeric
df['price'] = df['price'].str.replace('$', '').astype(float)
df['purchase_amount'] = df['purchase_amount'].str.replace('$', '').astype(float)

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Product Recommendation Analysis',
                style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px'}),
        html.P('Visualization of sales, ratings, and product performance data',
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px', 'marginBottom': '20px'})
    ]),
    # Dashboard introduction and storytelling section
    html.Div([
        html.H2('Dashboard Overview', style={'color': '#34495e', 'marginTop': '10px'}),
        html.P(
            "This dashboard provides a comprehensive analysis of e-commerce product sales, ratings, and user behaviors. "
            "Use the filters to explore trends by category and price range. Hover over charts for detailed tooltips. "
            "Key findings and insights are highlighted throughout the dashboard."
        , style={'fontSize': '16px', 'color': '#222', 'marginBottom': '15px'}),
        html.Ul([
            html.Li("Bar, pie, donut, line, and scatter charts are used to communicate different insights."),
            html.Li("Interactive filters and tooltips allow deep exploration of the data."),
            html.Li("Annotations and highlights guide you to key findings.")
        ], style={'fontSize': '15px', 'color': '#444', 'marginBottom': '20px'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '10px', 'margin': '15px'}),
    # Main content
    html.Div([
        # Left column - Filters
        html.Div([
            html.H3('Product Filters', style={'color': '#34495e', 'marginBottom': '15px'}),
            html.Label('Select Product Category', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': str(cat), 'value': cat} for cat in sorted(df['category'].unique())],
                multi=True,
                placeholder='Select one or more categories',
                style={'marginBottom': '25px'}
            ),
            
            html.Label('Price Range ($)', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.RangeSlider(
                id='price-range',
                min=int(df['price'].min()),
                max=int(df['price'].max()),
                step=10,
                marks={i: f'${i}' for i in range(int(df['price'].min()), int(df['price'].max())+1, 50)},
                value=[int(df['price'].min()), int(df['price'].max())],
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False
            ),
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
        
        # Right column - Charts
        html.Div([
            # First row - Key Metrics
            html.Div([
                html.Div([
                    html.H4('Key Metrics', style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': '15px'}),
                    html.Div(id='key-metrics', className='metrics-container')
                ], style={'width': '100%', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            ]),
            
            # Second row of charts
            html.Div([
                html.Div([
                    html.H4('Sales Distribution ($)', style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='purchase-dist')
                ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px'}),
                
                html.Div([
                    html.H4('Customer Ratings (1-5 Stars)', style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='rating-dist')
                ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
            
            # Third row of charts
            html.Div([
                html.Div([
                    html.H4('Product Categories', style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='category-dist')
                ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px'}),
                
                html.Div([
                    html.H4('Top Selling Products', style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='top-products')
                ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
            
            # Fourth row - time series
            html.Div([
                html.Div([
                    html.H4('Monthly Sales Trend', style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='time-series')
                ], style={'width': '100%', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px'}),
            ]),
        ], style={'width': '73%', 'padding': '20px'}),
    ], style={'display': 'flex', 'gap': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1'}),
])

# Callback for key metrics
@app.callback(
    Output('key-metrics', 'children'),
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_key_metrics(selected_categories, price_range):
    # Filter the data based on selections
    filtered_df = df.copy()
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                             (filtered_df['price'] <= price_range[1])]
    
    # Calculate metrics
    total_sales = filtered_df['purchase_amount'].sum()
    avg_rating = filtered_df['rating'].mean()
    unique_products = filtered_df['product_name'].nunique()
    top_category = filtered_df['category'].value_counts().idxmax() if not filtered_df.empty else 'N/A'
    
    # Create metric cards
    metrics = [
        html.Div([
            html.H3(f"${total_sales:,.2f}", style={'color': '#3498db', 'margin': '0'}),
            html.P("Total Sales", style={'margin': '0'})
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#eef8ff', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3(f"{avg_rating:.1f} ★", style={'color': '#2ecc71', 'margin': '0'}),
            html.P("Average Rating", style={'margin': '0'})
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#efffef', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3(f"{unique_products:,}", style={'color': '#e74c3c', 'margin': '0'}),
            html.P("Unique Products", style={'margin': '0'})
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#ffefef', 'borderRadius': '5px'}),
        
        html.Div([
            html.H3(top_category, style={'color': '#9b59b6', 'margin': '0'}),
            html.P("Top Category", style={'margin': '0'})
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f5eeff', 'borderRadius': '5px'}),
    ]
    
    return html.Div(metrics, style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '15px'})

# Callback for purchase amount distribution
@app.callback(
    Output('purchase-dist', 'figure'),
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_purchase_dist(selected_categories, price_range):
    # Filter the data based on selections
    filtered_df = df.copy()
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                             (filtered_df['price'] <= price_range[1])]
    
    fig = px.histogram(
        filtered_df,
        x='purchase_amount',
        nbins=30,
        labels={'purchase_amount': 'Purchase Amount ($)', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#3498db'],
        hover_data=['category', 'product_name', 'rating']  # Add more hover info
    )
    # Add a mean line
    mean_purchase = filtered_df['purchase_amount'].mean()
    fig.add_vline(x=mean_purchase, line_dash="dash", line_color="red",
                  annotation_text=f"Average: ${mean_purchase:.2f}",
                  annotation_position="top right")
    # Example annotation for storytelling
    if mean_purchase > 0:
        fig.add_annotation(x=mean_purchase, y=0, text="Average Purchase Value", showarrow=True, arrowhead=1)
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=30, r=30, b=30),
        xaxis_title='Purchase Amount ($)',
        yaxis_title='Number of Transactions',
        hovermode='closest',
        legend=dict(title='Legend', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color='black')
    return fig

# Callback for rating distribution
@app.callback(
    Output('rating-dist', 'figure'),
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_rating_dist(selected_categories, price_range):
    # Filter the data based on selections
    filtered_df = df.copy()
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                             (filtered_df['price'] <= price_range[1])]
    
    # Create a histogram for star ratings
    fig = px.histogram(
        filtered_df,
        x='rating',
        nbins=10,
        color_discrete_sequence=['#2ecc71'],
        labels={'rating': 'Rating (Stars)', 'count': 'Number of Products'},
        hover_data=['product_name', 'category', 'purchase_amount']
    )
    # Add star symbols
    fig.update_xaxes(tickvals=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                    ticktext=['1★', '1.5★', '2★', '2.5★', '3★', '3.5★', '4★', '4.5★', '5★'])
    # Add a mean line
    mean_rating = filtered_df['rating'].mean()
    fig.add_vline(x=mean_rating, line_dash="dash", line_color="red",
                  annotation_text=f"Average: {mean_rating:.1f}★",
                  annotation_position="top right")
    # Annotation for storytelling
    if mean_rating > 0:
        fig.add_annotation(x=mean_rating, y=0, text="Average Rating", showarrow=True, arrowhead=1)
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=30, r=30, b=30),
        xaxis_title='Product Rating (Stars)',
        yaxis_title='Number of Products',
        hovermode='closest',
        legend=dict(title='Legend', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color='black')
    return fig

# Callback for category distribution
@app.callback(
    Output('category-dist', 'figure'),
    [Input('price-range', 'value')]
)
def update_category_dist(price_range):
    # Filter the data based on price range
    filtered_df = df[(df['price'] >= price_range[0]) & 
                    (df['price'] <= price_range[1])]
    
    # Get category counts and sort them
    category_counts = filtered_df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    category_counts = category_counts.sort_values('count', ascending=False)
    
    # Create pie chart with better formatting
    fig = px.pie(
        category_counts,
        values='count',
        names='category',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.5,  # More pronounced donut chart
        hover_data=['count']
    )
    # Improve layout and add information
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1 if i==0 else 0 for i in range(len(category_counts))])
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=30, r=30, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        annotations=[dict(text='Product Categories', x=0.5, y=0.5, font_size=15, showarrow=False),
                     dict(text='Most popular category highlighted', x=0.5, y=0.9, font_size=12, showarrow=False)]
    )
    return fig

# Callback for top products
@app.callback(
    Output('top-products', 'figure'),
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_top_products(selected_categories, price_range):
    # Filter the data based on selections
    filtered_df = df.copy()
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                             (filtered_df['price'] <= price_range[1])]
    
    # Group by product name and calculate statistics
    product_stats = filtered_df.groupby('product_name').agg({
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
    # Add annotation for top product
    if not product_stats.empty:
        top_product = product_stats.index[0]
        fig.add_annotation(x=product_stats['purchase_amount'].iloc[0], y=top_product, text="Top Seller", showarrow=True, arrowhead=1, font=dict(color='red', size=12))
    # Update layout for better readability
    fig.update_layout(
        title=None,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=30, r=100, b=30),  # Extra right margin for star ratings
        height=500,
        xaxis=dict(title='Total Sales ($)'),
        yaxis=dict(title=None, categoryorder='total ascending'),
        legend_title='Product Category',
        barmode='stack',
        showlegend=False  # Too many categories can make legend cluttered
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color='black')
    return fig

# Callback for time series analysis
@app.callback(
    Output('time-series', 'figure'),
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_time_series(selected_categories, price_range):
    # Filter the data based on selections
    filtered_df = df.copy()
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                             (filtered_df['price'] <= price_range[1])]
    
    # Convert purchase_date to datetime
    filtered_df['purchase_date'] = pd.to_datetime(filtered_df['purchase_date'])
    
    # Group by month and calculate total sales
    filtered_df['month'] = filtered_df['purchase_date'].dt.to_period('M')
    monthly_sales = filtered_df.groupby('month').agg({
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
    # Add annotation for month with highest sales
    if not monthly_sales.empty:
        best_month = monthly_sales.iloc[monthly_sales['purchase_amount'].idxmax()]['month']
        best_sales = monthly_sales['purchase_amount'].max()
        fig.add_annotation(x=best_month, y=best_sales, text="Peak Month", showarrow=True, arrowhead=1, font=dict(color='green', size=12))
    # Update layout for better readability
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, l=30, r=30, b=30),
        hovermode="x unified",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(title=None, tickformat='%b %Y')
    )
    # Update y-axes titles
    fig.update_yaxes(title_text="Total Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from advanced_analytics import AdvancedAnalytics

# Load the most recent processed data
def load_processed_data():
    try:
        processed_dir = Path('../data/processed')
        if not processed_dir.exists():
            logger.error(f'Processed data directory not found: {processed_dir}')
            return None
        
        files = list(processed_dir.glob('processed_data_*.csv'))
        if not files:
            logger.error('No processed data files found')
            return None
            
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f'Loading processed data from {latest_file}')
        return pd.read_csv(latest_file)
    except Exception as e:
        logger.error(f'Error loading processed data: {str(e)}')
        return None

# Initialize the Dash app
# Initialize Dash app with custom styling
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
    ]
)

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Product Recommendation Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                background-color: #f5f5f5;
            }
            .dash-graph {
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                background-color: white;
                padding: 15px;
                margin: 10px;
            }
            .dash-tab-content {
                padding: 20px;
            }
            .dash-tab {
                padding: 10px 20px;
                margin: 5px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load processed data
df = load_processed_data()
if df is None:
    logger.error('Failed to load data. Dashboard may not function properly.')
    df = pd.DataFrame()  # Empty DataFrame as fallback and initialize analytics
data_path = '/Users/piyushpatel/Documents/data analytics/product_recommendation_system/data/processed/processed_data.csv'
df = pd.read_csv(data_path)
analytics = AdvancedAnalytics(data_path)

# Generate analytics data
user_profiles = analytics.create_user_profiles()
top_searches = analytics.analyze_search_patterns()
time_analysis = analytics.time_based_analysis()
popular_products = analytics.product_popularity_score()
trending_products = analytics.get_trending_products()

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Product Recommendation Dashboard', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        html.Div([
            html.P(f'Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   style={'textAlign': 'right', 'color': '#7f8c8d'})
        ])
    ]),
    
    # Main Tabs
    dcc.Tabs([
        # Product Analytics Tab
        dcc.Tab(label='Product Analytics', children=[
            html.Div([
                # Filters
                html.Div([
                    html.H3('Filters'),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': cat, 'value': cat} 
                                for cat in df['category'].unique()] if not df.empty else [],
                        placeholder='Select Category',
                        multi=True
                    )
                ], style={'margin': '20px'}),
                
                # Graphs
                html.Div([
                    html.Div([
                        dcc.Graph(id='category-distribution'),
                        dcc.Graph(id='price-distribution')
                    ], className='row'),
                    html.Div([
                        dcc.Graph(id='sales-trend'),
                        dcc.Graph(id='product-performance')
                    ], className='row')
                ])
            ], className='dash-tab-content')
        ]),
        
        # User Behavior Tab
        dcc.Tab(label='User Behavior', children=[
            html.Div([
                html.Div([
                    dcc.Graph(id='user-activity-heatmap'),
                    dcc.Graph(id='purchase-patterns')
                ], className='row'),
                html.Div([
                    dcc.Graph(id='rating-distribution'),
                    dcc.Graph(id='user-segments')
                ], className='row')
            ], className='dash-tab-content')
        ]),
        
        # Recommendations Tab
        dcc.Tab(label='Recommendations', children=[
            html.Div([
                html.Div([
                    html.H3('User-Based Recommendations'),
                    dcc.Dropdown(
                        id='user-select',
                        options=[{'label': f'User {u}', 'value': u} 
                                for u in df['user_id'].unique()] if not df.empty else [],
                        placeholder='Select User'
                    ),
                    dcc.Graph(id='user-recommendations')
                ]),
                html.Div([
                    html.H3('Similar Products'),
                    dcc.Dropdown(
                        id='product-select',
                        options=[{'label': f'Product {p}', 'value': p} 
                                for p in df['product_id'].unique()] if not df.empty else [],
                        placeholder='Select Product'
                    ),
                    dcc.Graph(id='similar-products')
                ])
            ], className='dash-tab-content')
        ])
    ])
])

# Callbacks for updating visualizations
@app.callback(
    [Output('total-products', 'children'),
     Output('avg-rating', 'children'),
     Output('total-sales', 'children')],
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_metrics(category, price_range):
    filtered_df = df.copy()
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) &
        (filtered_df['price'] <= price_range[1])
    ]
    
    return (
        len(filtered_df['product_id'].unique()),
        f"{filtered_df['rating'].mean():.2f}",
        f"${filtered_df['price'].sum():,.2f}"
    )

@app.callback(
    Output('category-distribution', 'figure'),
    [Input('price-range', 'value')]
)
def update_category_distribution(price_range):
    filtered_df = df[
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ]
    fig = px.pie(
        filtered_df,
        names='category',
        title='Product Category Distribution'
    )
    return fig

@app.callback(
    Output('price-distribution', 'figure'),
    [Input('category-filter', 'value')]
)
def update_price_distribution(category):
    filtered_df = df.copy()
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    fig = px.histogram(
        filtered_df,
        x='price',
        title='Price Distribution',
        nbins=50
    )
    return fig

@app.callback(
    Output('rating-history', 'figure'),
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_rating_history(category, price_range):
    filtered_df = df.copy()
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) &
        (filtered_df['price'] <= price_range[1])
    ]
    
    fig = px.line(
        filtered_df.groupby('date')['rating'].mean().reset_index(),
        x='date',
        y='rating',
        title='Average Rating Over Time'
    )
    return fig

@app.callback(
    Output('top-products', 'figure'),
    [Input('category-filter', 'value'),
     Input('price-range', 'value')]
)
def update_top_products(category, price_range):
    filtered_df = df.copy()
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) &
        (filtered_df['price'] <= price_range[1])
    ]
    
    top_products = (
        filtered_df.groupby('product_id')
        ['rating'].mean()
        .sort_values(ascending=False)
        .head(10)
    )
    
    fig = px.bar(
        x=top_products.index,
        y=top_products.values,
        title='Top 10 Products by Rating'
    )
    return fig

# Time Analysis Callbacks
@app.callback(
    [Output('daily-trends', 'figure'),
     Output('hourly-patterns', 'figure'),
     Output('weekly-patterns', 'figure')],
    [Input('category-filter', 'value')]
)
def update_time_analysis(category):
    filtered_data = time_analysis['daily_trends']
    if category:
        filtered_data = filtered_data[filtered_data['category'] == category]
    
    daily_fig = px.line(
        filtered_data,
        x='purchase_date',
        y='purchase_count',
        title='Daily Purchase Trends'
    )
    
    hourly_fig = px.bar(
        time_analysis['hourly_patterns'],
        x='hour',
        y='count',
        title='Purchases by Hour of Day'
    )
    
    weekly_fig = px.bar(
        time_analysis['weekly_patterns'],
        x='purchase_date',
        y='count',
        title='Purchases by Day of Week'
    )
    
    return daily_fig, hourly_fig, weekly_fig

# Product Analytics Callbacks
@app.callback(
    [Output('popular-products', 'figure'),
     Output('trending-products', 'figure'),
     Output('rating-distribution', 'figure')],
    [Input('category-filter', 'value')]
)
def update_product_analytics(category):
    filtered_popular = popular_products
    if category:
        filtered_popular = filtered_popular[filtered_popular['category'] == category]
    
    popular_fig = px.bar(
        filtered_popular.head(10),
        x='product_id',
        y='popularity_score',
        title='Top 10 Popular Products'
    )
    
    trending_fig = px.bar(
        trending_products.head(10),
        x='product_id',
        y='trend_score',
        title='Top 10 Trending Products'
    )
    
    rating_fig = px.histogram(
        df,
        x='rating',
        nbins=20,
        title='Rating Distribution'
    )
    
    return popular_fig, trending_fig, rating_fig

# User Behavior Callbacks
@app.callback(
    [Output('user-purchase-patterns', 'figure'),
     Output('search-trends', 'figure'),
     Output('category-preferences', 'figure')],
    [Input('category-filter', 'value')]
)
def update_user_behavior(category):
    filtered_users = user_profiles
    if category:
        filtered_users = filtered_users[filtered_users['category'].apply(lambda x: category in x)]
    
    patterns_fig = px.scatter(
        filtered_users,
        x='purchase_count',
        y='avg_spending',
        color='rating',
        title='User Purchase Patterns'
    )
    
    search_fig = px.bar(
        top_searches.head(10),
        x='term',
        y='importance',
        title='Top Search Terms'
    )
    
    preferences_fig = px.pie(
        df['category'].value_counts().reset_index(),
        values='count',
        names='category',
        title='Category Preferences'
    )
    
    return patterns_fig, search_fig, preferences_fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
