import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model from Hugging Face"""
    HF_USERNAME = "ssingh1404"  # Your username
    MODEL_NAME = "wellness-tourism-model"
    
    try:
        model_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/{MODEL_NAME}",
            filename="best_model.pkl",
            repo_type="model"
        )
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Wellness Tourism Package Predictor")
    st.markdown("### Predict customer purchase likelihood for wellness tourism packages")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check configuration.")
        return
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_encoded = 1 if gender == "Male" else 0
        
        occupation = st.selectbox(
            "Occupation", 
            ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )
        occupation_map = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
        occupation_encoded = occupation_map[occupation]
        
        designation = st.selectbox(
            "Designation", 
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        designation_map = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
        designation_encoded = designation_map[designation]
        
        marital_status = st.selectbox(
            "Marital Status", 
            ["Single", "Married", "Divorced", "Unmarried"]
        )
        marital_map = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
        marital_encoded = marital_map[marital_status]
        
        monthly_income = st.number_input(
            "Monthly Income", 
            min_value=1000, 
            max_value=100000, 
            value=25000,
            step=1000
        )
    
    with col2:
        st.subheader("Travel Preferences")
        
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        
        type_of_contact = st.selectbox(
            "Type of Contact", 
            ["Company Invited", "Self Enquiry"]
        )
        contact_encoded = 0 if type_of_contact == "Company Invited" else 1
        
        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting", 
            min_value=1, max_value=5, value=2
        )
        
        number_of_children_visiting = st.number_input(
            "Number of Children Visiting", 
            min_value=0, max_value=3, value=0
        )
        
        number_of_trips = st.number_input(
            "Average Trips per Year", 
            min_value=0, max_value=20, value=2
        )
        
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
        passport_encoded = 1 if passport == "Yes" else 0
        
        own_car = st.selectbox("Owns Car?", ["Yes", "No"])
        own_car_encoded = 1 if own_car == "Yes" else 0
    
    st.subheader("Sales Interaction Details")
    col3, col4 = st.columns(2)
    
    with col3:
        duration_of_pitch = st.slider(
            "Duration of Pitch (minutes)", 
            min_value=0, max_value=60, value=15
        )
        
        number_of_followups = st.slider(
            "Number of Follow-ups", 
            min_value=0, max_value=10, value=3
        )
    
    with col4:
        pitch_satisfaction_score = st.selectbox(
            "Pitch Satisfaction Score", 
            [1, 2, 3, 4, 5],
            index=2
        )
        
        product_pitched = st.selectbox(
            "Product Pitched", 
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        product_map = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
        product_encoded = product_map[product_pitched]
        
        preferred_property_star = st.selectbox(
            "Preferred Property Star Rating", 
            [3.0, 4.0, 5.0]
        )
    
    # Prediction button
    st.markdown("---")
    if st.button("Predict Purchase Likelihood", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [contact_encoded],
            'CityTier': [city_tier],
            'DurationOfPitch': [float(duration_of_pitch)],
            'Occupation': [occupation_encoded],
            'Gender': [gender_encoded],
            'NumberOfPersonVisiting': [number_of_person_visiting],
            'NumberOfFollowups': [float(number_of_followups)],
            'ProductPitched': [product_encoded],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_encoded],
            'NumberOfTrips': [float(number_of_trips)],
            'Passport': [passport_encoded],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'OwnCar': [own_car_encoded],
            'NumberOfChildrenVisiting': [float(number_of_children_visiting)],
            'Designation': [designation_encoded],
            'MonthlyIncome': [float(monthly_income)]
        })
        
        # Make prediction
        try:
            with st.spinner('Making prediction...'):
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Prediction", 
                    "Will Purchase" if prediction == 1 else "Won't Purchase"
                )
            
            with col_b:
                st.metric(
                    "Purchase Probability", 
                    f"{prediction_proba[1]:.2%}"
                )
            
            with col_c:
                st.metric(
                    "Confidence", 
                    f"{max(prediction_proba):.2%}"
                )
            
            # Status message
            if prediction == 1:
                st.success("High Likelihood: Customer is likely to purchase the wellness package")
            else:
                st.warning("Low Likelihood: Customer is unlikely to purchase the package")
            
            # Recommendations
            st.markdown("### Recommendations")
            if prediction == 1:
                st.info("""
                **Suggested Actions:**
                - Prioritize this lead for immediate follow-up
                - Prepare personalized package offers
                - Schedule consultation call
                """)
            else:
                st.info("""
                **Suggested Actions:**
                - Consider nurture campaigns
                - Gather more information about preferences
                - Re-engage after 2-3 months
                """)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
