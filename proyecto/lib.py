import yfinance as yf

# Obtener datos históricos de precios de acciones de Amazon
data = yf.download('AMZN', start='2014-01-01', end='2024-05-25')

# Guardar los datos en un archivo CSV
data.to_csv('amazon_stock_prices.csv')