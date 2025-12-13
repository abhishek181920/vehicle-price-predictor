import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-price {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .feature-importance {
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessor."""
    try:
        # Load the model dictionary and extract the model
        model_data = joblib.load('models/vehicle_price_predictor.pkl')
        model = model_data['model']  # Extract the model from the dictionary
        preprocessor = joblib.load('models/preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # Show full traceback for debugging
        return None, None

# Load model and preprocessor
model, preprocessor = load_model()

# App title and description
def main():
    st.title("🚗 Vehicle Price Predictor")
    st.markdown("### Predict the market value of your vehicle based on its specifications")
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app predicts vehicle prices using machine learning. 
        The model was trained on a dataset of various vehicles with their specifications.
        """)
        
        st.markdown("### Model Metrics")
        try:
            with open('models/metrics.json', 'r') as f:
                import json
                metrics = json.load(f)
                st.metric("R² Score", f"{metrics['r2']:.3f}")
                st.metric("RMSE", f"${metrics['rmse']:,.2f}")
                st.metric("MAE", f"${metrics['mae']:,.2f}")
        except:
            st.warning("Model metrics not available. Train the model first.")
    
    # Create form for input features
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vehicle Details")
            make = st.text_input("Make (e.g., Toyota, Ford)", "Toyota")
            model_name = st.text_input("Model", "Camry")
            year = st.number_input("Year", min_value=1950, max_value=datetime.now().year, value=2020)
            body = st.selectbox(
                "Body Type",
                ["Sedan", "SUV", "Truck", "Coupe", "Convertible", "Wagon", "Van", "Hatchback"]
            )
            
        with col2:
            st.subheader("Technical Specifications")
            engine = st.text_input("Engine", "2.5L 4-cylinder")
            transmission = st.selectbox(
                "Transmission",
                ["Automatic", "Manual", "CVT", "Semi-Automatic"]
            )
            drivetrain = st.selectbox(
                "Drivetrain",
                ["FWD", "RWD", "AWD", "4WD"]
            )
            fuel = st.selectbox(
                "Fuel Type",
                ["Gasoline", "Diesel", "Hybrid", "Electric", "Plug-in Hybrid"]
            )
            
        col3, col4 = st.columns(2)
        with col3:
            mileage = st.number_input("Mileage", min_value=0, value=30000)
            cylinders = st.selectbox(
                "Number of Cylinders",
                ["4", "6", "8", "10", "12", "Electric"]
            )
            
        with col4:
            doors = st.selectbox("Number of Doors", [2, 4])
            exterior_color = st.text_input("Exterior Color", "White")
            interior_color = st.text_input("Interior Color", "Black")
        
        # Submit button
        submitted = st.form_submit_button("Predict Price")
    
    if submitted and model is not None and preprocessor is not None:
        try:
            # Prepare input data with all required columns
            input_data = pd.DataFrame([{
                'make': make,
                'model': model_name,
                'year': year,
                'body': body,
                'engine': engine,
                'engine_size': float(engine.split('L')[0]) if engine and 'L' in engine.split()[0] else 2.0,  # Extract engine size
                'transmission': transmission,
                'drivetrain': drivetrain,
                'fuel': fuel,
                'mileage': mileage,
                'cylinders': 0 if cylinders == 'Electric' else int(cylinders),  # Handle 'Electric' as 0 cylinders
                'trim': 'Base',  # Default trim level
                'doors': doors,
                'exterior_color': exterior_color,
                'interior_color': interior_color,
                'vehicle_age': datetime.now().year - year
            }])
            
            # Ensure all expected columns are present
            expected_columns = ['make', 'model', 'year', 'body', 'engine', 'engine_size', 'transmission',
                              'drivetrain', 'fuel', 'mileage', 'cylinders', 'trim', 'doors', 'exterior_color',
                              'interior_color', 'vehicle_age']
            
            # Add any missing columns with default values
            for col in expected_columns:
                if col not in input_data.columns:
                    input_data[col] = None  # Or appropriate default value
            
            # Preprocess input data
            X = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Display prediction
            st.markdown("## Prediction Result")
            with st.container():
                st.markdown(
                    f"<div class='prediction-box'>"
                    f"<h3>Estimated Market Value</h3>"
                    f"<div class='prediction-price'>${prediction:,.2f}</div>"
                    f"<p>Based on the provided vehicle details</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Show feature importance if available
                try:
                    st.markdown("### How Features Affect the Price")
                    st.image("models/plots/feature_importance.png", 
                            caption="Feature Importance",
                            use_column_width=True)
                except:
                    pass
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Add some space at the bottom
    st.markdown("---")
    st.markdown("*Note: This is a prediction based on available data and may not reflect actual market value.*")

if __name__ == "__main__":
    if model is None or preprocessor is None:
        st.error("Model or preprocessor not found. Please train the model first.")
    else:
        main()
