import json
from pathlib import Path

import h5py
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


def _normalize_dtype(dtype_config):
    if isinstance(dtype_config, str):
        return dtype_config
    if isinstance(dtype_config, dict):
        return dtype_config.get("config", {}).get("name")
    return None


def _build_compatible_h5_model(model_path: Path):
    # Keras 3.x may write H5 configs with fields that older runtimes cannot deserialize.
    with h5py.File(model_path, "r") as model_file:
        raw_config = model_file.attrs["model_config"]

    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")

    model_config = json.loads(raw_config)
    sequential_config = model_config.get("config", {})
    layer_definitions = sequential_config.get("layers", [])

    input_shape = None
    input_name = None
    input_dtype = "float32"
    model = tf.keras.Sequential(name=sequential_config.get("name", model_path.stem))

    for layer_definition in layer_definitions:
        class_name = layer_definition.get("class_name")
        layer_config = layer_definition.get("config", {})

        if class_name == "InputLayer":
            batch_shape = layer_config.get("batch_shape") or layer_config.get("batch_input_shape")
            if not batch_shape or len(batch_shape) < 2:
                raise ValueError(f"Unsupported input shape in {model_path}.")

            input_shape = tuple(dimension for dimension in batch_shape[1:] if dimension is not None)
            input_name = layer_config.get("name")
            input_dtype = _normalize_dtype(layer_config.get("dtype")) or input_dtype
            continue

        if class_name != "Dense":
            raise ValueError(f"Unsupported layer '{class_name}' in {model_path}.")

        if input_shape is None:
            raise ValueError(f"Missing InputLayer metadata in {model_path}.")

        dense_kwargs = {
            "units": layer_config["units"],
            "activation": layer_config.get("activation"),
            "use_bias": layer_config.get("use_bias", True),
            "name": layer_config.get("name"),
        }
        dense_dtype = _normalize_dtype(layer_config.get("dtype"))
        if dense_dtype:
            dense_kwargs["dtype"] = dense_dtype

        if not model.layers:
            model.add(tf.keras.Input(shape=input_shape, name=input_name, dtype=input_dtype))

        dense_layer = tf.keras.layers.Dense(**dense_kwargs)
        dense_layer.trainable = layer_config.get("trainable", True)
        model.add(dense_layer)

    model.load_weights(model_path)
    return model


def load_model_with_fallback(model_path: Path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError):
        return _build_compatible_h5_model(model_path)


@st.cache_resource
def load_models():
    churn_model = load_model_with_fallback(CHURN_MODEL_PATH)
    salary_model = load_model_with_fallback(SALARY_MODEL_PATH)
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
