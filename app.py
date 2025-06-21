import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load model and dataset
model = joblib.load("model.pkl")
df = pd.read_csv(r"C:\Users\aryan\Downloads\archive (1)\Laptop_price.csv")  # Your dataset path

# Sidebar navigation
screen = st.sidebar.selectbox("Select Screen", ["💻 Prediction", "📊 Graph", "🗃️ Database"])

# Base price estimator
def estimate_base_price(brand, ram, storage):
    base_price = 20000
    if brand == "Apple":
        base_price += 30000
    elif brand in ["MSI", "Dell", "HP"]:
        base_price += 15000
    elif brand == "Lenovo":
        base_price += 10000
    else:
        base_price += 5000
    base_price += ram * 1000
    base_price += (storage // 128) * 500
    return max(base_price, 50000)

# 💻 Prediction Screen
if screen == "💻 Prediction":
    st.title("💻 Future Laptop Price Prediction")
    st.write("Fill in the specifications to predict laptop price.")

    brand = st.selectbox("Brand", df["Brand"].unique())
    processor = st.slider("Processor Speed (GHz)", 1.0, 5.0, 2.5)
    ram = st.selectbox("RAM Size (GB)", sorted(df["RAM_Size"].unique()))
    storage = st.selectbox("Storage Capacity (GB)", sorted(df["Storage_Capacity"].unique()))
    screen_size = st.slider("Screen Size (inches)", 10.0, 18.0, 13.3)
    weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0)

    if st.button("Predict Price"):
        input_df = pd.DataFrame([{
            "Brand": brand,
            "Processor_Speed": processor,
            "RAM_Size": ram,
            "Storage_Capacity": storage,
            "Screen_Size": screen_size,
            "Weight": weight
        }])

        # Display input summary
        st.subheader("📝 Laptop Specifications")
        st.write(f"**Brand:** {brand}")
        st.write(f"**Processor Speed:** {processor} GHz")
        st.write(f"**RAM:** {ram} GB")
        st.write(f"**Storage:** {storage} GB")
        st.write(f"**Screen Size:** {screen_size} inches")
        st.write(f"**Weight:** {weight} kg")

        # Predict price
        log_price = model.predict(input_df)[0]
        predicted_price = np.expm1(log_price)
        base_price = estimate_base_price(brand, ram, storage)

        st.subheader("💰 Estimated Laptop Price")
        st.success(f"₹{predicted_price:,.2f}")

        st.subheader("🏷️ Reference Market Price (Main Price)")
        st.info(f"₹{base_price:,.0f}")

# 📊 Graph Screen
elif screen == "📊 Graph":
    st.title("📈 Laptop Price Graphs")
    st.write("Visualize price trends by feature.")
    
    x_axis = st.selectbox("Select feature for X-axis", ["Brand", "RAM_Size", "Storage_Capacity", "Screen_Size", "Weight"])
    fig = px.box(df, x=x_axis, y="Price", color=x_axis, points="all", title=f"Price Distribution by {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

# 🗃️ Database Screen
elif screen == "🗃️ Database":
    st.title("🗃️ Laptop Dataset")
    st.dataframe(df)
