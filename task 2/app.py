import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="📈 Revenue Forecast Dashboard", layout="wide")
st.title("📊 Revenue Forecast with Polynomial Regression")

# === Load Excel ===
df = pd.read_excel(r"C:\Users\Malavi\Desktop\task 2\Ancillary Rev - 2015 - 2023.xlsx", header=2)
df['Flight Date'] = pd.to_datetime(df['Flight Date'])
df['Year'] = df['Flight Date'].dt.year
df['Month'] = df['Flight Date'].dt.to_period('M').dt.to_timestamp()

# === Filter Sidebar ===
pos_list = df['Point of Sale'].dropna().unique()
channel_list = df['Channel'].dropna().unique()
selected_pos = st.sidebar.selectbox("Filter by Point of Sale", options=np.append(["All"], pos_list))
selected_channel = st.sidebar.selectbox("Filter by Channel", options=np.append(["All"], channel_list))
view_mode = st.sidebar.radio("Forecast Mode", ["Yearly", "Monthly"])

if selected_pos != "All":
    df = df[df['Point of Sale'] == selected_pos]

if selected_channel != "All":
    df = df[df['Channel'] == selected_channel]

# === Top POS/Channel ===
top_pos = df.groupby('Point of Sale')['Sum of Revenue USD'].sum().idxmax()
top_pos_rev = df.groupby('Point of Sale')['Sum of Revenue USD'].sum().max()
top_channel = df.groupby('Channel')['Sum of Revenue USD'].sum().idxmax()
top_channel_rev = df.groupby('Channel')['Sum of Revenue USD'].sum().max()

col1, col2 = st.columns(2)
col1.metric("💰 Top Point of Sale", top_pos, f"${top_pos_rev:,.0f}")
col2.metric("🛍️ Top Channel", top_channel, f"${top_channel_rev:,.0f}")

# === Forecast Settings ===
degree = st.slider("Polynomial Degree", 1, 5, 2)
forecast_horizon = st.slider("Forecast Period", 1, 5, 3)

# === Prepare Data ===
if view_mode == "Yearly":
    group_data = df.groupby('Year')['Sum of Revenue USD'].sum().reset_index()
    X = group_data[['Year']]
    y = group_data['Sum of Revenue USD']
    future = pd.DataFrame({'Year': range(X['Year'].max() + 1, X['Year'].max() + forecast_horizon + 1)})
    label = 'Year'
else:
    group_data = df.groupby('Month')['Sum of Revenue USD'].sum().reset_index()
    group_data['MonthInt'] = np.arange(len(group_data))
    X = group_data[['MonthInt']]
    y = group_data['Sum of Revenue USD']
    future = pd.DataFrame({'MonthInt': np.arange(len(group_data), len(group_data) + forecast_horizon)})
    label = 'Month'

# === Fit Polynomial Model ===
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# === Forecasting ===
X_all = pd.concat([X, future])
X_all_poly = poly.transform(X_all)
y_pred_all = model.predict(X_all_poly)

# === Evaluation ===
y_train_pred = model.predict(poly.transform(X))
mae = mean_absolute_error(y, y_train_pred)
rmse = np.sqrt(mean_squared_error(y, y_train_pred))
r2 = r2_score(y, y_train_pred)

st.subheader("📉 Model Evaluation")
st.write(f"**MAE:** {mae:,.0f}")
st.write(f"**RMSE:** {rmse:,.0f}")
st.write(f"**R² Score:** {r2:.2f}")

# === Plot ===
st.subheader("📈 Forecast Plot")
fig = go.Figure()

if view_mode == "Yearly":
    fig.add_trace(go.Scatter(x=X['Year'], y=y, mode='markers+lines', name="Actual"))
    fig.add_trace(go.Scatter(x=X_all[label], y=y_pred_all, mode='lines+markers', name="Forecast"))
    x_label = "Year"
else:
    fig.add_trace(go.Scatter(x=group_data['Month'], y=y, mode='markers+lines', name="Actual"))
    forecast_months = pd.date_range(start=group_data['Month'].max() + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq='MS')
    fig.add_trace(go.Scatter(x=pd.concat([group_data['Month'], pd.Series(forecast_months)]), y=y_pred_all, mode='lines+markers', name="Forecast"))
    x_label = "Month"

fig.update_layout(xaxis_title=x_label, yaxis_title="Revenue (USD)", height=500)
st.plotly_chart(fig, use_container_width=True)

# === Forecast Table ===
st.subheader("📄 Forecast Table")
if view_mode == "Yearly":
    forecast_df = future.copy()
    forecast_df['Predicted Revenue'] = model.predict(poly.transform(future))
else:
    forecast_df = pd.DataFrame({'Month': forecast_months})
    forecast_df['Predicted Revenue'] = model.predict(poly.transform(future))

st.dataframe(forecast_df)

# === Download Forecast ===
buffer = BytesIO()
forecast_df.to_csv(buffer, index=False)
buffer.seek(0)
st.download_button("📥 Download Forecast CSV", buffer, file_name="polynomial_forecast.csv", mime="text/csv")
