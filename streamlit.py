import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from lime import lime_tabular

# Set the page title and description
st.title("Real Estate Price Predictor")
st.write("""
This app predicts a real estate price 
based on various property characteristics.
""")

# Load the pre-trained model
with open("models/RFmodel.pkl", "rb") as pkl:
    rf_model = pickle.load(pkl)


def plot_explainer(prediction_input, rf_model):
    # Load the training data
    with open("models/xtrain.pkl", "rb") as file:
        x_train = pickle.load(file)

    # create explainer object
    LIMEexplainer = lime_tabular.LimeTabularExplainer(
            training_data=x_train,
            feature_names=["year_sold", "property_tax", "insurance", "beds", "baths", "sqft", "year_built", "lot_size", 
                                    "basement", "popular", "recession", "property_age", "property_type_Bunglow", "property_type_Condo"],
            mode='regression'
    )
    # Generate explanation instance
    exp = LIMEexplainer.explain_instance(
        data_row=prediction_input.iloc[0],                 
        predict_fn=rf_model.predict,           # Model's prediction function
        num_features=14                      # Number of features to include in explanation
    )
    # Convert explanation to a matplotlib figure
    fig = exp.as_pyplot_figure()  
    # Get feature importances from the explanation
    importances = [x[1] for x in exp.as_list()]  # Feature importance values
    importances.reverse()

    # Annotate each bar with its corresponding importance value
    for i, importance in enumerate(importances, start=0):
        plt.text(
            importance,  # x-coordinate of the bar (importance value)
            i,  # y-coordinate (corresponding bar)
            f'{importance:.0f}',  # Display importance value 
            ha='center',  # Align text horizontally 
            va='center',  # Align text vertically 
            fontsize=10,  # Font size for the annotation
            color='black'  # Text color
        )
    return fig, exp.as_list()


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Real Estate Details")
    
    # create 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Year Sold 
        year_sold = st.slider("Select transaction year", value=2007, min_value=1993, max_value=2016)
        
        # Property Tax
        property_tax = st.number_input("Property Tax",value=467, max_value=4500, step=100)

        # Insurance
        insurance = st.number_input("Insurance",value=140, max_value=1400, step=100)

        # Beds
        beds = st.selectbox("Number of bedrooms", ["1", "2", "3", "4", "5"])

        # Baths
        baths = st.selectbox("Number of bathrooms", ["1", "2", "3", "4", "5", "6"])

    with col2:
        # sqft
        sqft = st.number_input("Sqft of the property", value = 2300, min_value=500, max_value=9000, step=100)

        # Year Built
        year_built = st.slider("Select built year", value= 1982, min_value=1880, max_value=2015)
    
    
        # Lot Size
        lot_size = st.number_input("Lot Size", min_value=0, max_value=500000, value=13000, step=1000)
        
        # Basement
        basement = st.selectbox("Basement", options=["1", "0"])
        
        # Property Type
        property_type = st.selectbox("Property Type", options=["Bunglow", "Condo"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Property Price")


# Handle the dummy variables to pass to the model
if submitted:
    # convert to integers
    year_sold = int(year_sold)
    property_tax = int(property_tax)
    insurance = int(insurance)
    beds = int(beds)
    baths = int(baths)
    sqft = int(sqft)
    year_built = int(year_built)
    lot_size = int(lot_size)
    basement =int(basement)
    if year_built > year_sold:
        st.write("The year built cannot greater than the year sold. Try again.")
        st.stop()

    # deal dummy feature
    property_type_Bunglow = 1 if property_type == "Bunglow" else 0
    property_type_Condo = 1 if property_type == "Condo" else 0
    
    popular = 1 if beds == 2 and baths == 2 else 0
    recession = 1 if (year_sold >= 2010) and (year_sold<=2013) else 0
    property_age = year_sold - year_built


    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = pd.DataFrame([[year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, 
                                     basement, popular, recession, property_age, property_type_Bunglow, property_type_Condo]], 
                           columns=["year_sold", "property_tax", "insurance", "beds", "baths", "sqft", "year_built", "lot_size", 
                                    "basement", "popular", "recession", "property_age", "property_type_Bunglow", "property_type_Condo"]
    )

    # Make prediction
    new_prediction = rf_model.predict(prediction_input)

    # Display result
    st.subheader("Prediction Result:")
    st.write(f"The predicted price is: ${int(new_prediction[0]):,}")

    fig, exp = plot_explainer(prediction_input, rf_model)
    # Display explanation in Streamlit
    st.subheader("LIME Explanation for Prediction")
    st.pyplot(fig)

    st.subheader("Feature Contributions:")
    st.table(pd.DataFrame(exp, columns=["Feature", "Importance"]))
    

st.write(
    """A Random Forest Regression model is used to predict your property price,
    the features used in this prediction are ranked by relative importance below."""
)
st.image("feature_importance.png")
