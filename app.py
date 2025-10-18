import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_style("whitegrid")
# To ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Force CPU (Streamlit Cloud has no GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- 1. Data Loading, Preprocessing, Training, and Model Saving/Loading ---
@st.cache_resource
def load_or_train_model(file_path='eCommerce_Customer_support_data.csv'):
    """Load saved model & preprocessor if available, else train new ones."""
    model_path = 'deepcsat_ann_model.keras'
    preprocessor_path = 'deepcsat_preprocessor.joblib'

    # 1️⃣ Try to load existing model & preprocessor
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        st.info("📦 Loading existing model and preprocessor...")
        model = load_model(model_path)
        preprocessor = joblib.load(preprocessor_path)

        df = pd.read_csv(file_path)
        categorical_features = ['channel_name', 'category', 'Sub-category', 
                                'Product_category', 'Agent Shift', 'Tenure Bucket']
        category_options = {}
        for col in categorical_features:
            category_options[col] = sorted([str(val) for val in df[col].fillna('Unknown').unique()])
        feature_columns = df.columns.tolist()
        return model, preprocessor, feature_columns, category_options

    # 2️⃣ Train new model if no saved model exists
    st.warning("⚙️ No saved model found — training a new one (first run only)...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Required file not found: {file_path}. Please upload it.")
        return None, None, None, None

    X = df.drop('CSAT Score', axis=1)
    y = df['CSAT Score'].values - 1

    X['Customer Remarks'] = X['Customer Remarks'].fillna('')
    X['connected_handling_time'] = X['connected_handling_time'].fillna(X['connected_handling_time'].median())

    X_dropped = X.drop(columns=[
        'Unique id', 'Order_id', 'order_date_time', 'Issue_reported at',
        'issue_responded', 'Survey_response_Date', 'Customer_City',
        'Agent_name', 'Supervisor', 'Manager'
    ])

    categorical_features = ['channel_name', 'category', 'Sub-category',
                            'Product_category', 'Agent Shift', 'Tenure Bucket']

    for col in categorical_features:
        X_dropped[col] = X_dropped[col].fillna('Unknown')

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['connected_handling_time']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('text', TfidfVectorizer(max_features=5000, stop_words='english'), 'Customer Remarks')
    ])

    X_train, _, y_train, _ = train_test_split(X_dropped, y, test_size=0.2, stratify=y, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train).toarray()
    y_train_categorical = to_categorical(y_train)

    # Class weights to handle imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    st.info(f"Using class weights: {class_weights}")

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_processed.shape[1],), name='Hidden_Layer_1'),
        Dropout(0.3),
        Dense(128, activation='relu', name='Hidden_Layer_2'),
        Dropout(0.3),
        Dense(64, activation='relu', name='Hidden_Layer_3'),
        Dense(5, activation='softmax', name='Output_Layer')
    ], name='DeepCSAT_ANN')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train_processed,
        y_train_categorical,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        class_weight=class_weights
    )

    # Save for future runs
    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)

    category_options = {}
    for col in categorical_features:
        category_options[col] = sorted([str(val) for val in df[col].fillna('Unknown').unique()])

    st.success("✅ Model training complete and saved for future use.")
    return model, preprocessor, X_dropped.columns.tolist(), category_options


# Load or train model safely
model, preprocessor, feature_columns, category_options = load_or_train_model()

# --- 2. Streamlit UI Design ---
st.set_page_config(
    page_title="DeepCSAT E-commerce Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛍️ DeepCSAT Score Predictor (E-commerce)")
st.subheader("Predicting Customer Satisfaction using Deep Learning on Interaction Data")

if model is not None:
    st.markdown("---")
    
    # Input Form
    with st.form("csat_prediction_form"):
        st.header("Enter Interaction Details")
        
        col1, col2, col3 = st.columns(3)

        # Column 1
        with col1:
            st.markdown("#### Agent & Channel Context")
            if 'channel_name' in category_options:
                input_channel_name = st.selectbox("Channel Name", options=category_options['channel_name'])
            if 'Tenure Bucket' in category_options:
                input_tenure_bucket = st.selectbox("Agent Tenure Bucket", options=category_options['Tenure Bucket'])
            if 'Agent Shift' in category_options:
                input_agent_shift = st.selectbox("Agent Shift", options=category_options['Agent Shift'])
            
        # Column 2
        with col2:
            st.markdown("#### Issue & Product Details")
            if 'category' in category_options:
                input_category = st.selectbox("Issue Category", options=category_options['category'])
            if 'Sub-category' in category_options:
                input_subcategory = st.selectbox("Issue Sub-category", options=category_options['Sub-category'])
            if 'Product_category' in category_options:
                input_product_category = st.selectbox("Product Category", options=category_options['Product_category'])

        # Column 3
        with col3:
            st.markdown("#### Time & Feedback")
            input_connected_handling_time = st.number_input(
                "Connected Handling Time (seconds)",
                min_value=0, max_value=3600, value=300, step=10
            )

        input_customer_remarks = st.text_area(
            "Customer Remarks/Feedback (Crucial for Prediction)",
            value="I am extremely frustrated! My package is delayed by a week and the agent couldn't help me resolve the issue. Bad service.",
            height=150
        )
        
        submitted = st.form_submit_button("Predict CSAT Score")

    # Prediction Logic
    if submitted:
        input_data = pd.DataFrame({
            'channel_name': [input_channel_name],
            'category': [input_category],
            'Sub-category': [input_subcategory],
            'Product_category': [input_product_category],
            'Agent Shift': [input_agent_shift],
            'Tenure Bucket': [input_tenure_bucket],
            'connected_handling_time': [input_connected_handling_time],
            'Customer Remarks': [input_customer_remarks]
        })

        with st.spinner('Analyzing feedback and predicting CSAT...'):
            try:
                input_processed = preprocessor.transform(input_data).toarray()
                prediction_probs = model.predict(input_processed)[0]
                predicted_class = np.argmax(prediction_probs) + 1

                st.success("CSAT Prediction Complete!")
                
                col_result, col_probs = st.columns([1, 2])
                
                with col_result:
                    st.metric("Predicted CSAT Score", f"{predicted_class} / 5", delta="Proactive Insight")
                    if predicted_class <= 2:
                        st.error("🚨 ALERT: Predicted Low CSAT (1-2)")
                        st.markdown("**Action:** Immediate supervisor follow-up required for service recovery.")
                    elif predicted_class == 5:
                        st.balloons()
                        st.success("⭐ High Satisfaction Predicted!")
                    else:
                        st.warning("Needs monitoring.")

                with col_probs:
                    st.markdown("#### Probability Distribution")
                    prob_df = pd.DataFrame({'CSAT Score': [str(i) for i in range(1,6)], 'Probability': prediction_probs})
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x='CSAT Score', y='Probability', data=prob_df, palette='Spectral', ax=ax)
                    ax.set_title("Model Confidence Across CSAT Categories")
                    ax.set_ylim(0,1)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Prediction failed. Please check input data consistency. Error: {e}")
