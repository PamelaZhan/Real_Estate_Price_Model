# real_estate_price_application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://real-estate-price-predictor-app.streamlit.app/)

This application predicts the fair transaction price of a property before it is sold in a small county in New York state. It uses a dataset of transaction prices from previously sold properties to make these predictions. The model aims to predict transaction prices with an average error of less than $70,000.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details of real estate such as property_tax, insurance, beds, baths, Bunglow/Condo, and other relevant factors.
- Real-time prediction of property price based on the trained Random Forest model. Mean Absolute Error (MAE) < $70,000
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained on the **Real Estate dataset**, a dataset of transaction prices for previously sold properties on the market. It includes features like:
- Price
- Year_sold
- Property_tax
- Insurance
- Beds
- Baths
- Sqft
- Year_built
- Lot_size
- Basement
- Property_type(Bunglow/Condo)

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).
- **LIME**: Local Interpretable Model-Agnostic Explanations, for explaining the model prediction.

## Model
The Random Forest model is trained using the real estate dataset. It applies preprocessing steps like adding domain features and encoding categorical variables. 

#### Thank you for using the Real Estate Price Application! Feel free to share your feedback.
