[Live Deployment: deepchurn-analytics.streamlit.app](https://deepchurn-analytics.streamlit.app/)

# ANN Customer Churn And Salary Prediction

## Project Overview
This project is a machine learning web application built with Streamlit. It combines two TensorFlow models in one interface:

- A customer churn classification model that predicts whether a bank customer is likely to leave the bank.
- An estimated salary regression model that predicts salary from the remaining customer profile details.

The application uses sidebar navigation so both prediction pages are available from a single `app.py` file.

## Application Pages

### 1. Customer Churn Prediction
This page uses the saved classification model in `model.h5` to predict churn probability.

User inputs:
- Geography
- Gender
- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Output:
- Churn probability
- Likely to churn or not likely to churn message

### 2. Estimated Salary Prediction
This page uses the saved regression model in `regression_model.h5` to predict estimated salary.

User inputs:
- Geography
- Gender
- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Exited

Output:
- Predicted estimated salary

## Main Features
- Single Streamlit application with two model pages
- Clean sidebar navigation
- TensorFlow model loading with caching
- Data preprocessing recreated from the dataset for reliable deployment
- Classification and regression workflows in one project
- Streamlit Cloud deployment-ready dependency file

## Tech Stack
- Python
- Streamlit
- TensorFlow
- Scikit-learn
- Pandas
- NumPy

## Dataset
The project uses `Churn_Modelling.csv`, which contains bank customer information such as:

- Demographics
- Account activity
- Product usage
- Balance
- Salary
- Churn label

This dataset is used to rebuild the preprocessing pipeline inside the app so that both models receive the correct input format.

## How The App Works
1. The app loads the churn model from `model.h5`.
2. The app loads the salary regression model from `regression_model.h5`.
3. It reads `Churn_Modelling.csv`.
4. It applies label encoding for gender and one-hot encoding for geography.
5. It rebuilds separate feature scaling logic for the churn model and salary model.
6. It takes user input from the Streamlit interface and generates predictions.

## Project Structure
```text
ANN_customer_churn_prediction/
|-- app.py
|-- README.md
|-- requirements.txt
|-- Churn_Modelling.csv
|-- model.h5
|-- regression_model.h5
|-- experiments.ipynb
|-- salaryregression.ipynb
|-- prediction.ipynb
|-- label_encoder_gender.pkl
|-- onehot_encoder_geo.pkl
|-- scaler.pkl
|-- logs/
`-- regressionlogs/
```

## Local Setup
### 1. Clone the repository
```bash
git clone <your-repository-url>
cd ANN_customer_churn_prediction
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment
To deploy this project on Streamlit Community Cloud:

1. Push the project to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Click `New app`.
4. Select your repository, branch, and main file path as `app.py`.
5. Make sure `requirements.txt` is present in the root directory.
6. Deploy the app.

Recommended:
- Use Python 3.11 in Streamlit Cloud advanced settings to match the local environment used in this project.
- Keep `model.h5`, `regression_model.h5`, and `Churn_Modelling.csv` in the repository root.

## Example Use Cases
- Predict whether a customer is likely to churn
- Demonstrate deployment of deep learning models with Streamlit
- Compare classification and regression in one project
- Build a simple end-to-end machine learning portfolio project

## Notes
- The churn result is a probability-based prediction.
- The salary result is an estimate and should not be treated as an exact real-world salary.
- The application rebuilds preprocessing from the dataset during runtime to keep both models aligned with the expected feature columns.

## Future Improvements
- Add model evaluation metrics directly to the app
- Add charts and data insights
- Save user predictions to a file or database
- Add better validation and default value suggestions
- Deploy with custom styling and branding

## Conclusion
This project demonstrates how to deploy multiple machine learning models in a single Streamlit application. It is useful for learning model serving, data preprocessing, UI building with Streamlit, and cloud deployment for practical ML applications.
