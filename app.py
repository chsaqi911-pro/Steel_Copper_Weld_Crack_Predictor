import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Weld Crack Predictor",
    page_icon="🔧",
    layout="wide"
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================
@st.cache_resource
def load_model():
    # Make sure weld_crack_prediction_model.pkl is in your repository root directory
    return joblib.load("weld_crack_prediction_model.pkl")

model = load_model()

# ============================================================
# TITLE
# ============================================================
st.title("🔧 Weld Crack Predictor")

st.markdown(
    """
    ### Machine Learning-Based Weld Crack Prediction

    Enter the welding parameters below to predict whether the weld
    is likely to contain a crack.

    """
)

# ============================================================
# MODEL INFORMATION
# ============================================================
with st.expander("ℹ️ About the Model"):
    st.write(
        """
        The trained model is a weighted Logistic Regression model.

        The machine learning pipeline consists of:
        **Input → Median Imputation → StandardScaler → Weighted Logistic Regression → Prediction**

        SHAP (SHapley Additive exPlanations) is used to explain
        the contribution of each welding parameter to the prediction.
        """
    )

# ============================================================
# INPUT METHOD
# ============================================================
st.subheader("⚙️ Welding Parameters")

input_method = st.radio(
    "Select input method:",
    ["Slider", "Manual Numeric Input"],
    horizontal=True
)

# ============================================================
# INPUT RANGES (Corrected feature key names matching the .pkl)
# ============================================================
feature_info = {
    "power (W)": {
        "label": "power (W)",
        "min": 500.0,
        "max": 1600.0,
        "default": 1050.0,
        "step": 10.0
    },
    "welding_speed (m/min)": {
        "label": "Welding speed (m/min)",
        "min": 0.30,
        "max": 1.80,
        "default": 1.00,
        "step": 0.01
    },
    "heat_dissipated ": {
        "label": "heat dissipated ",
        "min": 10.0,
        "max": 80.0,
        "default": 45.0,
        "step": 1.0
    },
    "gas_flow_rate (l/min)": {
        "label": "Gas flow rate (L/min)",
        "min": 3.0,
        "max": 30.0,
        "default": 15.0,
        "step": 0.5
    },
    "focal_position (mm)": {
        "label": "Focal position (mm)",
        "min": -6.0,
        "max": 6.0,
        "default": 0.0,
        "step": 0.1
    },
    "angular_position (°)": {
        "label": "Angular position (°)",
        "min": -45.0,
        "max": 45.0,
        "default": 0.0,
        "step": 1.0
    },
    "material_thickness (mm)": {
        "label": "Material thickness (mm)",
        "min": 0.20,
        "max": 1.00,
        "default": 0.60,
        "step": 0.01
    },
    "cross_section_weld_positon (mm)": {
        "label": "Cross-section weld position (mm)",
        "min": 0.0,
        "max": 50.0,
        "default": 20.0,
        "step": 0.5
    },
    "gap": {
        "label": "Gap",
        "min": 0.0,
        "max": 150.0,
        "default": 65.0,
        "step": 0.5
    }
}

# ============================================================
# INPUT GENERATION
# ============================================================
col1, col2 = st.columns(2)
user_inputs = {}

# We programmatically build UI controls using our defined mappings
for i, (feat_key, info) in enumerate(feature_info.items()):
    target_col = col1 if i < 5 else col2
    
    with target_col:
        if input_method == "Slider":
            val = st.slider(
                info["label"],
                min_value=info["min"],
                max_value=info["max"],
                value=info["default"],
                step=info["step"],
                key=f"slide_{feat_key}"
            )
        else:
            val = st.number_input(
                info["label"],
                min_value=info["min"],
                max_value=info["max"],
                value=info["default"],
                step=info["step"],
                key=f"num_{feat_key}"
            )
        user_inputs[feat_key] = val

# ============================================================
# CREATE INPUT DATAFRAME WITH MATCHING PKL KEYS
# ============================================================
# Ensures the column order aligns perfectly with your pipeline
input_data = pd.DataFrame([user_inputs])

# ============================================================
# PREDICTION BUTTON
# ============================================================
st.divider()
predict_button = st.button(
    "🔍 Predict Weld Condition",
    type="primary",
    use_container_width=True
)

# ============================================================
# PREDICTION EXECUTION
# ============================================================
if predict_button:
    try:
        # Predict using the Pipeline steps directly or through the pipeline wrapper
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Identify predicted class indices
        classes_list = list(model.classes_)
        predicted_class_index = classes_list.index(prediction)
        confidence = probabilities[predicted_class_index] * 100

        prediction_text = "Crack" if prediction == 1 else "No Crack"

        # Display result layout
        st.divider()
        st.subheader("🎯 Prediction Result")
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(label="Prediction", value=prediction_text)

        with result_col2:
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        if prediction == 1:
            st.error(f"⚠️ Prediction: **Crack**\n\nConfidence Score: **{confidence:.2f}%**")
        else:
            st.success(f"✅ Prediction: **No Crack**\n\nConfidence Score: **{confidence:.2f}%**")

        # ====================================================
        # SHAP EXPLANATION
        # ====================================================
        st.divider()
        st.subheader("📊 SHAP Explanation")
        st.markdown(
            """
            The SHAP plot explains the contribution of each welding parameter to the prediction.
            - **Positive SHAP values:** push the prediction toward **Crack**.
            - **Negative SHAP values:** push the prediction toward **No Crack**.
            """
        )

        try:
            # Safely deconstruct elements from pipeline
            imputer = model.named_steps["imputer"]
            scaler = model.named_steps["scaler"]
            classifier = model.named_steps["model"]

            # Step-by-step transform for the pipeline dependencies
            X_imputed = imputer.transform(input_data)
            X_scaled = scaler.transform(X_imputed)

            # Generate linear background model explanation
            explainer = shap.LinearExplainer(classifier, X_scaled)
            shap_values = explainer(X_scaled)

            # Re-map names back onto the chart using human-readable UI names
            shap_values.feature_names = [feature_info[col]["label"] for col in input_data.columns]

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], max_display=9, show=False)
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as shap_error:
            st.warning("The prediction was successful, but the SHAP explanation could not be generated.")
            with st.expander("SHAP Technical Information"):
                st.exception(shap_error)

        # ====================================================
        # DISPLAY RAW PARAMETERS
        # ====================================================
        st.divider()
        st.subheader("📋 Input Parameters Used")
        
        # Display inputs visually matching original layout format
        display_data = pd.DataFrame({
            "Parameter": [feature_info[col]["label"] for col in input_data.columns],
            "Value": input_data.iloc[0].values
        })
        st.dataframe(display_data, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error("An error occurred while making the prediction.")
        with st.expander("Technical Information"):
            st.exception(error)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Weld Crack Predictor | Machine Learning-Based Weld Defect Prediction")
