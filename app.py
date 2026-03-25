from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


st.set_page_config(page_title="Bank Prediction Suite", page_icon=":bank:", layout="wide")

DATA_PATH = Path("data/Churn_Modelling.csv")
CHURN_MODEL_PATH = Path("models/model.h5")
SALARY_MODEL_PATH = Path("models/regression_model.h5")
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


@st.cache_resource
def load_models():
    churn_model = tf.keras.models.load_model(CHURN_MODEL_PATH, compile=False)
    salary_model = tf.keras.models.load_model(SALARY_MODEL_PATH, compile=False)
    return churn_model, salary_model


@st.cache_resource
def build_preprocessors():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=DROP_COLUMNS)

    label_encoder_gender = LabelEncoder()
    data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])

    onehot_encoder_geo = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    geo_encoded = onehot_encoder_geo.fit_transform(data[["Geography"]])
    geo_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])

    data = pd.concat(
        [
            data.drop(columns=["Geography"]).reset_index(drop=True),
            pd.DataFrame(geo_encoded, columns=geo_columns),
        ],
        axis=1,
    )

    train_indices, _ = train_test_split(data.index, test_size=0.20, random_state=42)

    churn_features = data.drop(columns=["Exited"])
    salary_features = data.drop(columns=["EstimatedSalary"])

    churn_scaler = StandardScaler()
    churn_scaler.fit(churn_features.loc[train_indices])

    salary_scaler = StandardScaler()
    salary_scaler.fit(salary_features.loc[train_indices])

    return label_encoder_gender, onehot_encoder_geo, churn_scaler, salary_scaler


def render_customer_inputs(
    key_prefix: str,
    geography_options,
    gender_options,
    include_salary: bool = False,
    include_exited: bool = False,
):
    left_col, right_col = st.columns(2)

    with left_col:
        geography = st.selectbox("Geography", geography_options, key=f"{key_prefix}_geography")
        gender = st.selectbox("Gender", gender_options, key=f"{key_prefix}_gender")
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            step=1,
            key=f"{key_prefix}_credit_score",
        )
        age = st.slider("Age", min_value=18, max_value=92, value=35, key=f"{key_prefix}_age")
        tenure = st.slider("Tenure", min_value=0, max_value=10, value=5, key=f"{key_prefix}_tenure")

    with right_col:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key=f"{key_prefix}_balance",
        )
        num_of_products = st.slider(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=1,
            key=f"{key_prefix}_num_of_products",
        )
        has_cr_card = st.selectbox(
            "Has Credit Card",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
            key=f"{key_prefix}_has_cr_card",
        )
        is_active_member = st.selectbox(
            "Is Active Member",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
            key=f"{key_prefix}_is_active_member",
        )

        estimated_salary = None
        exited = None

        if include_salary:
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                key=f"{key_prefix}_estimated_salary",
            )

        if include_exited:
            exited = st.selectbox(
                "Exited",
                [0, 1],
                format_func=lambda value: "No" if value == 0 else "Yes",
                key=f"{key_prefix}_exited",
            )

    return {
        "Geography": geography,
        "Gender": gender,
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Exited": exited,
    }


def build_common_features(inputs, label_encoder_gender, onehot_encoder_geo):
    base_features = pd.DataFrame(
        {
            "CreditScore": [inputs["CreditScore"]],
            "Gender": [label_encoder_gender.transform([inputs["Gender"]])[0]],
            "Age": [inputs["Age"]],
            "Tenure": [inputs["Tenure"]],
            "Balance": [inputs["Balance"]],
            "NumOfProducts": [inputs["NumOfProducts"]],
            "HasCrCard": [inputs["HasCrCard"]],
            "IsActiveMember": [inputs["IsActiveMember"]],
        }
    )

    geo_encoded = onehot_encoder_geo.transform(pd.DataFrame({"Geography": [inputs["Geography"]]}))
    geo_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])
    geo_features = pd.DataFrame(geo_encoded, columns=geo_columns)

    return pd.concat([base_features.reset_index(drop=True), geo_features], axis=1)


def prepare_churn_features(inputs, label_encoder_gender, onehot_encoder_geo, churn_scaler):
    features = build_common_features(inputs, label_encoder_gender, onehot_encoder_geo)
    features["EstimatedSalary"] = inputs["EstimatedSalary"]
    return features.reindex(columns=churn_scaler.feature_names_in_, fill_value=0.0)


def prepare_salary_features(inputs, label_encoder_gender, onehot_encoder_geo, salary_scaler):
    features = build_common_features(inputs, label_encoder_gender, onehot_encoder_geo)
    features["Exited"] = inputs["Exited"]
    return features.reindex(columns=salary_scaler.feature_names_in_, fill_value=0.0)


def show_churn_page(churn_model, label_encoder_gender, onehot_encoder_geo, churn_scaler):
    st.title("Customer Churn Prediction")
    st.caption("Use the classification model to estimate whether a customer is likely to churn.")

    with st.form("churn_prediction_form"):
        inputs = render_customer_inputs(
            "churn",
            onehot_encoder_geo.categories_[0].tolist(),
            label_encoder_gender.classes_.tolist(),
            include_salary=True,
        )
        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        return

    features = prepare_churn_features(inputs, label_encoder_gender, onehot_encoder_geo, churn_scaler)
    scaled_features = churn_scaler.transform(features)
    churn_probability = float(churn_model.predict(scaled_features, verbose=0)[0][0])

    st.metric("Churn Probability", f"{churn_probability:.2%}")
    if churn_probability > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")


def show_salary_page(salary_model, label_encoder_gender, onehot_encoder_geo, salary_scaler):
    st.title("Estimated Salary Prediction")
    st.caption("Use the regression model to estimate salary from the other customer details.")

    with st.form("salary_prediction_form"):
        inputs = render_customer_inputs(
            "salary",
            onehot_encoder_geo.categories_[0].tolist(),
            label_encoder_gender.classes_.tolist(),
            include_exited=True,
        )
        submitted = st.form_submit_button("Predict Salary")

    if not submitted:
        return

    features = prepare_salary_features(inputs, label_encoder_gender, onehot_encoder_geo, salary_scaler)
    scaled_features = salary_scaler.transform(features)
    predicted_salary = float(salary_model.predict(scaled_features, verbose=0)[0][0])

    st.metric("Predicted Estimated Salary", f"${predicted_salary:,.2f}")
    st.info("This regression output should be treated as an estimate, not an exact salary value.")


def main():
    churn_model, salary_model = load_models()
    label_encoder_gender, onehot_encoder_geo, churn_scaler, salary_scaler = build_preprocessors()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Customer Churn Prediction", "Estimated Salary Prediction"],
    )

    st.sidebar.markdown("One app, two model pages.")

    if page == "Customer Churn Prediction":
        show_churn_page(churn_model, label_encoder_gender, onehot_encoder_geo, churn_scaler)
    else:
        show_salary_page(salary_model, label_encoder_gender, onehot_encoder_geo, salary_scaler)


if __name__ == "__main__":
    main()
