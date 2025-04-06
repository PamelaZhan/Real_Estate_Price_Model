
from src.data.load_dataset import load_and_preprocess_data
from src.features.build_features import create_indicator_dummy_vars
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_decision_tree
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
import pandas as pd

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/real_estate.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    x, y = create_indicator_dummy_vars(df)

    # Train the dicision tree regression model
    model, x_test, y_test = train_RFmodel(x, y)



    # Evaluate the model
    mae = evaluate_model(model, x_test, y_test)    
    if mae<70000:
        print(f"Successful! The mean absolute error is {mae}. The model achieved the goal: error under 70000.")
        # Plot
        plot_correlation_heatmap(pd.merge(y,x, left_index=True, right_index=True))
        plot_feature_importance(model, x)
    else:
        print(f"The mean absolute error is over 70000. It is not fit.")

 
