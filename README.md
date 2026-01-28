# Laptop-Price-Prediction

Laptop Price Prediction 💻
This project is a machine learning solution designed to predict laptop prices based on various specifications such as RAM, CPU, Storage, and more. It utilizes the powerful XGBoost Regressor to achieve high-precision results.

🚀 Project Overview
The project features a fully automated pipeline from data preprocessing to model evaluation. The model is optimized to capture complex relationships in the dataset, ensuring reliable price estimations.

📊 Model Performance
The model has been rigorously tested and shows exceptional performance metrics:
1. R2 Score (Test): $0.9996$ — Indicates that the model explains $99.96\%$ of the price variance.
2. 5-Fold CV Mean R2: $0.9995$ — Confirms the model's consistency across different data subsets.
3. RMSE: $196.928$ — On average, the predictions are within approximately $197$ units of the actual price.

📂 Project Structure
As seen in the project directory:

1. app/: Contains app.py for deploying the model as a web application.

2. data/: Stores raw datasets and generated test predictions.

3. models/: Holds the serialized final_model.pkl for future use.

4. notebooks/: Jupyter notebooks for Exploratory Data Analysis (EDA) and Modeling.

5. results/: Automatically generated evaluation plots and metrics.

6. src/: Core Python scripts for training (train.py) and preprocessing (preprocessing.py).

🛠️ Installation & Usage
1. Clone the repository and install dependencies:

pip install -r requirements.txt

2. Train the model and generate results:
python src/train.py

📈 Visual Insights
After running the training script, the following visualizations are generated in the results/ folder:

1. Feature Importance: Displays which laptop specifications (e.g., RAM, GPU) impact the price the most.

2. Residual Plot: Analyzes the distribution of errors to ensure no underlying patterns are missed.

3. Predicted vs Actual: A visual comparison showing how closely the model's predictions align with real market prices.