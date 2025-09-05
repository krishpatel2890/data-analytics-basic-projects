
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Load and preprocess data
df = pd.read_csv("supermart_grocery_sales.csv")

df['Sale'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

df.dropna(subset=['Sale', 'Profit', 'Discount', 'Order Date'], inplace=True)

df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day_name()
df['DiscountFlag'] = df['Discount'].apply(lambda x: 'Yes' if x > 0 else 'No')

groupable_columns = ['City', 'Region', 'Category', 'Sub Category', 'Month', 'Day']

# Linear models
model_discount = LinearRegression()
model_discount.fit(df[['Discount', 'Profit']], df['Sale'])

month_sales = df.groupby('Month')['Sale'].sum().reset_index()
model_month = LinearRegression()
model_month.fit(month_sales[['Month']], month_sales['Sale'])

# Dash app
app = dash.Dash(__name__)
app.title = "Supermarket Sales Dashboard"

app.layout = html.Div([
    html.H1("🛒 Supermarket Grocery Sales Analytics with Prediction"),

    html.Label("Select Category for Chart:"),
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': col, 'value': col} for col in groupable_columns],
        value='Region',
        clearable=False
    ),

    html.Div([
        dcc.Graph(id='bar-chart', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='pie-chart', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='line-chart', style={'width': '33%', 'display': 'inline-block'}),
    ]),

    html.H2("📦 Top Subcategories by Sales"),
    dcc.Graph(
        figure=px.bar(
            df.groupby('Sub Category')['Sale'].sum().sort_values(ascending=False).head(10).reset_index(),
            x='Sub Category', y='Sale', title="Top 10 Subcategories by Sales"
        )
    ),

    html.H2("👥 Top 10 Customers by Sales"),
    dcc.Graph(
        figure=px.bar(
            df.groupby('Customer Name')['Sale'].sum().sort_values(ascending=False).head(10).reset_index(),
            x='Customer Name', y='Sale', title="Top Customers"
        ) if 'Customer Name' in df.columns else px.bar(title="Customer data not available")
    ),

    html.H2("📆 Sales by Day of the Week"),
    dcc.Graph(
        figure=px.bar(
            df.groupby('Day')['Sale'].sum().reset_index(),
            x='Day', y='Sale', title="Sales Trend by Day of Week"
        )
    ),

    # html.H2("🎯 Sales With vs Without Discount"),
    # dcc.Graph(
    #     figure=px.bar(
    #         df.groupby('DiscountFlag')['Sale'].sum().reset_index(),
    #         x='DiscountFlag', y='Sale', color='DiscountFlag',
    #         title="Sales with vs without Discount"
    #     )
    # ),

    html.H2("🔮 Predict Sales Based on Discount and Profit"),
    html.Div([
        html.Label("Enter Discount (%)"),
        dcc.Input(id='input-discount', type='number', value=10, min=0, max=50, step=1),
        html.Label("Enter Profit (₹)"),
        dcc.Input(id='input-profit', type='number', value=500, min=0, step=50),
        html.Div(id='sales-prediction-output', style={'marginTop': '10px', 'fontSize': '20px', 'color': 'green'})
    ], style={'marginBottom': '40px'}),

    html.H2("📅 Predict Next Month's Sales"),
    html.Div(id='month-prediction-output', style={'fontSize': '20px', 'color': 'blue'})
])

# Update charts when category changes
@app.callback(
    [Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('line-chart', 'figure')],
    Input('category-dropdown', 'value')
)
def update_charts(category):
    grouped = df.groupby(category).agg({'Sale': 'sum', 'Profit': 'sum'}).reset_index()

    bar_fig = px.bar(grouped, x=category, y='Sale', title=f"Sales by {category}")
    pie_fig = px.pie(grouped, names=category, values='Sale', title=f"Sales Share by {category}")
    line_fig = px.line(grouped, x=category, y='Sale', title=f"Sales Trend by {category}")

    return bar_fig, pie_fig, line_fig

# Predict Sales from discount + profit
@app.callback(
    Output('sales-prediction-output', 'children'),
    [Input('input-discount', 'value'),
     Input('input-profit', 'value')]
)
def predict_sales(discount, profit):
    if discount is None or profit is None:
        return "Please enter values for discount and profit."

    input_df = pd.DataFrame({'Discount': [discount], 'Profit': [profit]})
    predicted_sale = model_discount.predict(input_df)[0]
    return f"Predicted Sales Amount: ₹ {predicted_sale:.2f}"

# Predict next month's sales
@app.callback(
    Output('month-prediction-output', 'children'),
    Input('category-dropdown', 'value')  # dummy trigger
)
def predict_next_month(category):
    next_month = int(df['Month'].max()) + 1 if df['Month'].max() < 12 else 1
    next_month_df = pd.DataFrame({'Month': [next_month]})
    pred_sale = model_month.predict(next_month_df)[0]
    return f"Predicted Sales for Month {next_month}: ₹ {pred_sale:.2f}"

if __name__ == '__main__':
    app.run(debug=True)
