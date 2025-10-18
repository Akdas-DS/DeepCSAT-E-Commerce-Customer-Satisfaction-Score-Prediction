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
from sklearn.utils.class_weight import compute_class_weight # New Import for handling imbalance
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
# To ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Data Loading, Preprocessing, Training, and Model Saving/Loading ---

@st.cache_resource
def load_and_train_assets(file_path='eCommerce_Customer_support_data.csv'):
    """
    Loads the training data, preprocesses it, trains the DeepCSAT model using 
    class weights to handle imbalance, and returns the assets.
    This function uses st.cache_resource to run only once.
    """
    st.info("Training model and fitting preprocessor (runs only once)... This may take a moment.")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Required file not found: {file_path}. Please ensure the data file is present.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

    # --- Preprocessing & Feature Engineering ---
    X = df.drop('CSAT Score', axis=1)
    # Convert CSAT 1-5 to 0-4 for Keras
    y = df['CSAT Score'].values - 1 

    X['Customer Remarks'] = X['Customer Remarks'].fillna('')
    numerical_features = ['connected_handling_time'] 
    X['connected_handling_time'] = X['connected_handling_time'].fillna(X['connected_handling_time'].median())
    
    # Drop identifiers and columns not available during real-time prediction
    X_dropped = X.drop(columns=['Unique id', 'Order_id', 'order_date_time', 
                                'Issue_reported at', 'issue_responded', 
                                'Survey_response_Date', 'Customer_City', 
                                'Agent_name', 'Supervisor', 'Manager'])

    categorical_features = ['channel_name', 'category', 'Sub-category', 
                            'Product_category', 'Agent Shift', 'Tenure Bucket']
    
    for col in categorical_features:
        X_dropped[col] = X_dropped[col].fillna('Unknown')

    text_feature = 'Customer Remarks'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('text', TfidfVectorizer(max_features=5000, stop_words='english'), text_feature)
        ],
        remainder='drop' 
    )

    # Train/Test Split
    X_train, _, y_train, _ = train_test_split(
        X_dropped, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train).toarray()
    y_train_categorical = to_categorical(y_train)

    input_shape = X_train_processed.shape[1]
    output_classes = 5
    
    # --- Calculate Class Weights to address Imbalance ---
    # Classes are 0, 1, 2, 3, 4 (representing CSAT 1 to 5)
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    # Convert to a dictionary for Keras: {class_index: weight_value}
    class_weights = dict(zip(classes, weights))
    st.info(f"Using class weights to improve low CSAT prediction: {class_weights}")


    # --- Deep Learning Model Development (ANN) ---
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,), name='Hidden_Layer_1'),
        Dropout(0.3),
        Dense(128, activation='relu', name='Hidden_Layer_2'),
        Dropout(0.3),
        Dense(64, activation='relu', name='Hidden_Layer_3'),
        Dense(output_classes, activation='softmax', name='Output_Layer')
    ], name='DeepCSAT_ANN')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train_processed, 
        y_train_categorical,
        epochs=10, # Reduced epochs for faster app startup
        batch_size=32,
        validation_split=0.1,
        verbose=0, # Suppress verbose output in Streamlit
        class_weight=class_weights # <-- Applied Class Weights Here
    )

    # Save model in .keras format and preprocessor (optional in this setup, but good practice)
    model.save('deepcsat_ann_model.keras')
    joblib.dump(preprocessor, 'deepcsat_preprocessor.joblib')
    st.success("Model training complete! Ready for prediction.")

    # Determine unique categorical values for Streamlit selectboxes
    category_options = {}
    for col in categorical_features:
        unique_values = df[col].fillna('Unknown').unique()
        category_options[col] = sorted([str(val) for val in unique_values if val is not None and val != ''])
            
    return model, preprocessor, X_dropped.columns.tolist(), category_options

model, preprocessor, feature_columns, category_options = load_and_train_assets()

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
    
    # --- Input Form ---
    with st.form("csat_prediction_form"):
        st.header("Enter Interaction Details")
        
        col1, col2, col3 = st.columns(3)

        # Column 1: Channel & Agent Data
        with col1:
            st.markdown("#### Agent & Channel Context")
            # Ensure 'channel_name' is available in category_options
            if 'channel_name' in category_options:
                input_channel_name = st.selectbox(
                    "Channel Name",
                    options=category_options['channel_name']
                )
            # Ensure 'Tenure Bucket' is available
            if 'Tenure Bucket' in category_options:
                input_tenure_bucket = st.selectbox(
                    "Agent Tenure Bucket",
                    options=category_options['Tenure Bucket']
                )
            # Ensure 'Agent Shift' is available
            if 'Agent Shift' in category_options:
                input_agent_shift = st.selectbox(
                    "Agent Shift",
                    options=category_options['Agent Shift']
                )
            
        # Column 2: Issue & Product Details
        with col2:
            st.markdown("#### Issue & Product Details")
            if 'category' in category_options:
                input_category = st.selectbox(
                    "Issue Category",
                    options=category_options['category']
                )
            if 'Sub-category' in category_options:
                input_subcategory = st.selectbox(
                    "Issue Sub-category",
                    options=category_options['Sub-category']
                )
            if 'Product_category' in category_options:
                input_product_category = st.selectbox(
                    "Product Category",
                    options=category_options['Product_category']
                )

        # Column 3: Time & Remarks
        with col3:
            st.markdown("#### Time & Feedback")
            input_connected_handling_time = st.number_input(
                "Connected Handling Time (seconds)",
                min_value=0,
                max_value=3600, 
                value=300,
                step=10
            )

        # Full-width text input for Customer Remarks (the most important feature)
        # Suggesting input that might lead to a low CSAT score for testing
        input_customer_remarks = st.text_area(
            "Customer Remarks/Feedback (Crucial for Prediction)",
            value="I am extremely frustrated! My package is delayed by a week and the agent couldn't help me resolve the issue. Bad service.",
            height=150
        )
        
        # Prediction button
        submitted = st.form_submit_button("Predict CSAT Score")

    # --- 3. Prediction Logic ---
    if submitted:
        # 1. Create a DataFrame from the inputs
        # Use the specific features dropped in the training step
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
                # 2. Apply the fitted preprocessor
                input_processed = preprocessor.transform(input_data)
                
                # Convert sparse to dense array for Keras
                input_dense = input_processed.toarray()

                # 3. Make Prediction
                prediction_probs = model.predict(input_dense)[0]
                predicted_class = np.argmax(prediction_probs) + 1 # Convert 0-4 to 1-5

                # 4. Display Results
                st.success("CSAT Prediction Complete!")
                
                col_result, col_probs = st.columns([1, 2])
                
                with col_result:
                    st.metric(
                        label="Predicted CSAT Score",
                        value=f"{predicted_class} / 5",
                        delta="Proactive Insight"
                    )
                    
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
                    
                    # Create a DataFrame for probability display
                    prob_df = pd.DataFrame({
                        'CSAT Score': [f'{i}' for i in range(1, 6)],
                        'Probability': prediction_probs
                    })
                    
                    # Plot the probabilities
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x='CSAT Score', y='Probability', data=prob_df, palette='Spectral', ax=ax)
                    ax.set_title("Model Confidence Across CSAT Categories")
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Prediction failed. Please check input data consistency. Error: {e}")


