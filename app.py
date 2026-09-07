```python
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

    **The application provides:**

    - Prediction: **Crack / No Crack**
    - Confidence Score of the prediction
    - SHAP-based explanation of the prediction
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

        **Input → Median Imputation → StandardScaler → 
        Weighted Logistic Regression → Prediction**

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
# INPUT RANGES
# ============================================================
#
# These are broad working ranges based on the statistics stored
# inside the trained StandardScaler.
#
# They are NOT claimed to be the exact min/max values of the
# original training dataset.
#
# If the original training dataset is available, these ranges
# can be replaced by its actual min/max values.
# ============================================================

feature_info = {

    "Dataset power (W)": {
        "min": 500.0,
        "max": 1600.0,
        "default": 1050.0,
        "step": 10.0
    },

    "welding_speed (m/min)": {
        "min": 0.30,
        "max": 1.80,
        "default": 1.00,
        "step": 0.01
    },

    "heat_dissipated": {
        "min": 10.0,
        "max": 80.0,
        "default": 45.0,
        "step": 1.0
    },

    "gas_flow_rate (l/min)": {
        "min": 3.0,
        "max": 30.0,
        "default": 15.0,
        "step": 0.5
    },

    "focal_position (mm)": {
        "min": -6.0,
        "max": 6.0,
        "default": 0.0,
        "step": 0.1
    },

    "angular_position (°)": {
        "min": -45.0,
        "max": 45.0,
        "default": 0.0,
        "step": 1.0
    },

    "material_thickness (mm)": {
        "min": 0.20,
        "max": 1.00,
        "default": 0.60,
        "step": 0.01
    },

    "cross_section_weld_positon (mm)": {
        "min": 0.0,
        "max": 50.0,
        "default": 20.0,
        "step": 0.5
    },

    "gap": {
        "min": 0.0,
        "max": 150.0,
        "default": 65.0,
        "step": 0.5
    }
}


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# SLIDER INPUT
# ============================================================

if input_method == "Slider":

    with col1:

        dataset_power = st.slider(
            "Dataset power (W)",
            min_value=feature_info["Dataset power (W)"]["min"],
            max_value=feature_info["Dataset power (W)"]["max"],
            value=feature_info["Dataset power (W)"]["default"],
            step=feature_info["Dataset power (W)"]["step"]
        )

        welding_speed = st.slider(
            "Welding speed (m/min)",
            min_value=feature_info["welding_speed (m/min)"]["min"],
            max_value=feature_info["welding_speed (m/min)"]["max"],
            value=feature_info["welding_speed (m/min)"]["default"],
            step=feature_info["welding_speed (m/min)"]["step"]
        )

        heat_dissipated = st.slider(
            "Heat dissipated",
            min_value=feature_info["heat_dissipated"]["min"],
            max_value=feature_info["heat_dissipated"]["max"],
            value=feature_info["heat_dissipated"]["default"],
            step=feature_info["heat_dissipated"]["step"]
        )

        gas_flow_rate = st.slider(
            "Gas flow rate (L/min)",
            min_value=feature_info["gas_flow_rate (l/min)"]["min"],
            max_value=feature_info["gas_flow_rate (l/min)"]["max"],
            value=feature_info["gas_flow_rate (l/min)"]["default"],
            step=feature_info["gas_flow_rate (l/min)"]["step"]
        )

        focal_position = st.slider(
            "Focal position (mm)",
            min_value=feature_info["focal_position (mm)"]["min"],
            max_value=feature_info["focal_position (mm)"]["max"],
            value=feature_info["focal_position (mm)"]["default"],
            step=feature_info["focal_position (mm)"]["step"]
        )

    with col2:

        angular_position = st.slider(
            "Angular position (°)",
            min_value=feature_info["angular_position (°)"]["min"],
            max_value=feature_info["angular_position (°)"]["max"],
            value=feature_info["angular_position (°)"]["default"],
            step=feature_info["angular_position (°)"]["step"]
        )

        material_thickness = st.slider(
            "Material thickness (mm)",
            min_value=feature_info["material_thickness (mm)"]["min"],
            max_value=feature_info["material_thickness (mm)"]["max"],
            value=feature_info["material_thickness (mm)"]["default"],
            step=feature_info["material_thickness (mm)"]["step"]
        )

        cross_section_weld_position = st.slider(
            "Cross-section weld position (mm)",
            min_value=feature_info["cross_section_weld_positon (mm)"]["min"],
            max_value=feature_info["cross_section_weld_positon (mm)"]["max"],
            value=feature_info["cross_section_weld_positon (mm)"]["default"],
            step=feature_info["cross_section_weld_positon (mm)"]["step"]
        )

        gap = st.slider(
            "Gap",
            min_value=feature_info["gap"]["min"],
            max_value=feature_info["gap"]["max"],
            value=feature_info["gap"]["default"],
            step=feature_info["gap"]["step"]
        )


# ============================================================
# MANUAL NUMERIC INPUT
# ============================================================

else:

    with col1:

        dataset_power = st.number_input(
            "Dataset power (W)",
            value=1050.0,
            step=10.0
        )

        welding_speed = st.number_input(
            "Welding speed (m/min)",
            value=1.00,
            step=0.01
        )

        heat_dissipated = st.number_input(
            "Heat dissipated",
            value=45.0,
            step=1.0
        )

        gas_flow_rate = st.number_input(
            "Gas flow rate (L/min)",
            value=15.0,
            step=0.5
        )

        focal_position = st.number_input(
            "Focal position (mm)",
            value=0.0,
            step=0.1
        )

    with col2:

        angular_position = st.number_input(
            "Angular position (°)",
            value=0.0,
            step=1.0
        )

        material_thickness = st.number_input(
            "Material thickness (mm)",
            value=0.60,
            step=0.01
        )

        cross_section_weld_position = st.number_input(
            "Cross-section weld position (mm)",
            value=20.0,
            step=0.5
        )

        gap = st.number_input(
            "Gap",
            value=65.0,
            step=0.5
        )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "Dataset power (W)": [dataset_power],

    "welding_speed (m/min)": [welding_speed],

    "heat_dissipated": [heat_dissipated],

    "gas_flow_rate (l/min)": [gas_flow_rate],

    "focal_position (mm)": [focal_position],

    "angular_position (°)": [angular_position],

    "material_thickness (mm)": [material_thickness],

    "cross_section_weld_positon (mm)": [
        cross_section_weld_position
    ],

    "gap": [gap]
})


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
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]


        # ----------------------------------------------------
        # IDENTIFY PREDICTED CLASS
        # ----------------------------------------------------

        predicted_class_index = list(
            model.classes_
        ).index(prediction)

        confidence = (
            probabilities[predicted_class_index] * 100
        )


        # ----------------------------------------------------
        # CONVERT CLASS TO ENGINEERING OUTPUT
        # ----------------------------------------------------

        if prediction == 1:

            prediction_text = "Crack"

        else:

            prediction_text = "No Crack"


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                label="Prediction",
                value=prediction_text
            )


        with result_col2:

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )


        # ----------------------------------------------------
        # RESULT MESSAGE
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                f"⚠️ Prediction: **Crack**\n\n"
                f"Confidence Score: **{confidence:.2f}%**"
            )

        else:

            st.success(
                f"✅ Prediction: **No Crack**\n\n"
                f"Confidence Score: **{confidence:.2f}%**"
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.subheader("📊 SHAP Explanation")

        st.markdown(
            """
            The SHAP plot explains the contribution of each
            welding parameter to the model prediction.

            - **Positive SHAP values:** push the prediction toward
              **Crack**.
            - **Negative SHAP values:** push the prediction toward
              **No Crack**.
            """
        )


        try:

            # ------------------------------------------------
            # Extract pipeline components
            # ------------------------------------------------

            imputer = model.named_steps["imputer"]

            scaler = model.named_steps["scaler"]

            classifier = model.named_steps["model"]


            # ------------------------------------------------
            # Apply training preprocessing
            # ------------------------------------------------

            X_imputed = imputer.transform(
                input_data
            )

            X_scaled = scaler.transform(
                X_imputed
            )


            # ------------------------------------------------
            # SHAP Linear Explainer
            # ------------------------------------------------

            explainer = shap.LinearExplainer(
                classifier,
                X_scaled
            )


            shap_values = explainer(
                X_scaled
            )


            # ------------------------------------------------
            # Feature names
            # ------------------------------------------------

            shap_values.feature_names = list(
                input_data.columns
            )


            # ------------------------------------------------
            # SHAP Waterfall Plot
            # ------------------------------------------------

            fig = plt.figure(
                figsize=(10, 7)
            )

            shap.plots.waterfall(
                shap_values[0],
                max_display=9,
                show=False
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        except Exception as shap_error:

            st.warning(
                "The prediction was successful, but the "
                "SHAP explanation could not be generated."
            )

            with st.expander("SHAP Technical Information"):

                st.code(
                    str(shap_error)
                )


        # ====================================================
        # INPUT PARAMETERS
        # ====================================================

        st.divider()

        st.subheader("📋 Input Parameters Used")

        display_data = input_data.T.reset_index()

        display_data.columns = [
            "Parameter",
            "Value"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


    except Exception as error:

        st.error(
            "An error occurred while making the prediction."
        )

        with st.expander("Technical Information"):

            st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Weld Crack Predictor | Machine Learning-Based "
    "Weld Defect Prediction"
)
```
