import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and feature names
model = joblib.load('agriculture_yield_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Crop types from the dataset
CROP_TYPES = ['Cotton', 'Carrot', 'Sugarcane', 'Tomato', 'Soybean', 'Rice', 
              'Maize', 'Wheat', 'Potato', 'Barley']

# Irrigation types from the dataset
IRRIGATION_TYPES = ['Sprinkler', 'Manual', 'Flood', 'Rain-fed', 'Drip']

# Soil types from the dataset
SOIL_TYPES = ['Loamy', 'Peaty', 'Silty', 'Clay', 'Sandy']

# Seasons from the dataset
SEASONS = ['Kharif', 'Zaid', 'Rabi']

def predict_yield(input_data):
    """Make prediction using the trained model"""
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Get prediction
    prediction = model.predict(input_df)
    
    return prediction[0]

def main():
    st.title("Agricultural Yield Prediction")
    st.write("""
    This app predicts crop yield based on farm characteristics and practices.
    """)
    
    # Sidebar with user input features
    st.sidebar.header('User Input Parameters')
    
    def user_input_features():
        crop_type = st.sidebar.selectbox('Crop Type', CROP_TYPES)
        farm_area = st.sidebar.number_input('Farm Area (acres)', min_value=0.0, value=100.0)
        irrigation_type = st.sidebar.selectbox('Irrigation Type', IRRIGATION_TYPES)
        fertilizer_used = st.sidebar.number_input('Fertilizer Used (tons)', min_value=0.0, value=5.0)
        pesticide_used = st.sidebar.number_input('Pesticide Used (kg)', min_value=0.0, value=2.0)
        soil_type = st.sidebar.selectbox('Soil Type', SOIL_TYPES)
        season = st.sidebar.selectbox('Season', SEASONS)
        water_usage = st.sidebar.number_input('Water Usage (cubic meters)', min_value=0.0, value=50000.0)
        
        data = {
            'Farm_Area(acres)': farm_area,
            'Crop_Type': crop_type,
            'Irrigation_Type': irrigation_type,
            'Fertilizer_Used(tons)': fertilizer_used,
            'Pesticide_Used(kg)': pesticide_used,
            'Soil_Type': soil_type,
            'Season': season,
            'Water_Usage(cubic meters)': water_usage
        }
        
        return data
    
    input_data = user_input_features()
    
    # Display user inputs
    st.subheader('User Input Parameters')
    input_df = pd.DataFrame([input_data])
    st.write(input_df)
    
    # Prediction
    if st.button('Predict Yield'):
        prediction = predict_yield(input_data)
        st.subheader('Predicted Yield')
        st.write(f"{prediction:.2f} tons")
    
    # Model insights
    st.subheader('Model Insights')
    
    # Feature importance (if available)
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        st.write("### Feature Importance")
        importances = model.named_steps['model'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        st.pyplot(plt)
    else:
        st.write("The selected model doesn't provide feature importance.")

if __name__ == '__main__':
    main()