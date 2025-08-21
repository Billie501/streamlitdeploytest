# app.py
import streamlit as st
import pandas as pd
import time
import json
import io
from datetime import datetime
import spacy

# Import your custom extractor
from extractors import EnsembleVotingExtractor

# =========================
# SpaCy Model Loader
# =========================
@st.cache_resource
def load_spacy_model():
    """
    Loads the spaCy model with fallback to blank model if download fails.
    Compatible with Streamlit Cloud restrictions.
    """
    import spacy
    
    try:
        # Try to load the model
        return spacy.load("en_core_web_sm")
    except OSError:
        # Model not found, try to create a blank model with some basic NLP capabilities
        st.warning("⚠️ spaCy model 'en_core_web_sm' not available. Using basic English model.")
        st.info("For full functionality, include the model in requirements.txt at build time.")
        
        try:
            # Create a blank English model
            nlp = spacy.blank("en")
            
            # Add basic pipeline components that don't require training
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            
            return nlp
            
        except Exception as e:
            st.error(f"Failed to create fallback model: {str(e)}")
            st.info("The app will continue with limited NLP functionality.")
            # Return a very basic blank model
            return spacy.blank("en")

nlp = load_spacy_model()

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="ML Entity Extraction Pipeline",
    page_icon="🤖",
    layout="wide"
)

# =========================
# Session State Init
# =========================
if "ensemble" not in st.session_state:
    st.session_state.ensemble = None
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "results_df" not in st.session_state:
    st.session_state.results_df = None

# Add this diagnostic section to your app.py to identify the root cause

def diagnose_model_performance():
    """Diagnostic function to identify why accuracy is poor on Streamlit"""
    
    st.header("🔍 Model Performance Diagnostics")
    
    # Check 1: SpaCy Model Version and Components
    if st.button("Check spaCy Model Status"):
        nlp = load_spacy_model()
        
        st.subheader("SpaCy Model Information:")
        st.write(f"**Model Name:** {nlp.meta.get('name', 'Unknown')}")
        st.write(f"**Model Version:** {nlp.meta.get('version', 'Unknown')}")
        st.write(f"**Language:** {nlp.meta.get('lang', 'Unknown')}")
        st.write(f"**Pipeline Components:** {nlp.pipe_names}")
        
        # Check if it's a blank model (which would explain poor performance)
        if not nlp.pipe_names or 'ner' not in nlp.pipe_names:
            st.error("⚠️ ISSUE FOUND: Using blank spaCy model without NER component!")
            st.info("This explains the poor accuracy. The model has no trained NER capabilities.")
        
        # Test the model on a simple example
        st.subheader("Model Test:")
        test_text = "John Smith works at Microsoft in Seattle and was injured on January 15, 2024."
        doc = nlp(test_text)
        
        st.write(f"**Test Text:** {test_text}")
        st.write(f"**Entities Found:** {[(ent.text, ent.label_) for ent in doc.ents]}")
        
        if not doc.ents:
            st.error("⚠️ ISSUE FOUND: spaCy model is not extracting any entities from test text!")
    
    # Check 2: Compare Environment Differences
    st.subheader("Environment Comparison:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Google Colab (High Accuracy):**")
        st.write("- Full spaCy model with all components")
        st.write("- More memory and processing power")
        st.write("- Stable file system for model loading")
        st.write("- Likely using spaCy 3.4+ with full NER")
    
    with col2:
        st.write("**Streamlit Cloud (Poor Accuracy):**")
        st.write("- Potentially blank/limited spaCy model")
        st.write("- Memory/CPU constraints")
        st.write("- Permission restrictions")
        st.write("- May be missing trained model components")
    
    # Check 3: Test Your Ensemble Components
    if st.session_state.ensemble:
        st.subheader("Ensemble Component Test:")
        test_text = st.text_area("Enter test text:", value="John Smith reported an incident at the warehouse on March 15th at 2:30 PM.")
        
        if st.button("Test Ensemble"):
            try:
                result, breakdown = st.session_state.ensemble.extract_with_voting(test_text)
                st.write("**Extraction Result:**")
                st.json(result)
                st.write("**Model Breakdown:**")
                st.json(breakdown)
                
                # Check if results are mostly empty
                non_empty_fields = sum(1 for v in result.values() if v and str(v).strip())
                total_fields = len(result)
                accuracy_estimate = (non_empty_fields / total_fields) * 100 if total_fields > 0 else 0
                
                if accuracy_estimate < 30:
                    st.error(f"⚠️ ISSUE FOUND: Only {accuracy_estimate:.1f}% of fields extracted!")
                    st.info("This suggests the underlying models are not working properly.")
                
            except Exception as e:
                st.error(f"⚠️ ISSUE FOUND: Ensemble extraction failed: {str(e)}")

  

# =========================
# Header
# =========================
st.title("🤖 Real-time ML Entity Extraction Pipeline")
st.markdown("Multi-Model Ensemble with Voting for unstructured data classification")

diagnose_model_performance()  

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Configuration")
batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=500, value=100)
show_intermediate = st.sidebar.checkbox("Show Intermediate Results", value=True)
show_model_breakdown = st.sidebar.checkbox("Show Model Breakdown", value=False)

# =========================
# Layout Columns
# =========================
col1, col2 = st.columns([2, 1])

# -------------------------
# Column 1 - Upload & Train
# -------------------------
with col1:
    st.header("Data Upload & Processing")

    uploaded_file = st.file_uploader(
        "Upload your CSV file with unstructured data",
        type=["csv"],
        help="CSV must contain a 'text' column"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ CSV must contain a 'text' column")
            st.stop()

        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.info(f"📊 Loaded {len(df)} rows and {len(df.columns)} columns")

        # --- Training Section ---
        st.subheader("Model Training")
        if st.button("🎯 Train Ensemble Models", type="primary"):
            with st.spinner("Training ensemble models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Initialize ensemble
                st.session_state.ensemble = EnsembleVotingExtractor()

                training_steps = ["SpaCy NER", "Hybrid Extractor", "Template ML", "Advanced Ensemble"]
                train_texts = df["text"].tolist()[: min(1000, len(df))]
                train_labels = [{}] * len(train_texts)  # placeholder

                for i, step in enumerate(training_steps):
                    status_text.text(f"Training {step}...")
                    progress_bar.progress((i + 1) / len(training_steps))

                    if step == "SpaCy NER":
                        st.session_state.ensemble.spacy_extractor.train(train_texts, train_labels)
                    elif step == "Hybrid Extractor":
                        st.session_state.ensemble.hybrid_extractor.train_ml_components(train_texts, train_labels)
                    elif step == "Template ML":
                        st.session_state.ensemble.template_extractor.train_classifiers(train_texts, train_labels)
                    elif step == "Advanced Ensemble":
                        st.session_state.ensemble.advanced_extractor.train(train_texts, train_labels)

                st.session_state.is_trained = True
                status_text.text("✅ All models trained successfully!")
                st.success("🎉 Training completed!")

        # --- Processing Section ---
        if st.session_state.is_trained and st.session_state.ensemble:
            st.subheader("Real-time Processing")

            if st.button("🚀 Start Processing", type="primary"):
                results = []
                total_rows = len(df)

                main_progress = st.progress(0)
                status_container = st.empty()
                metrics_container = st.container()
                results_container = st.empty()

                with metrics_container:
                    col_metrics = st.columns(4)
                    col_metrics[0].metric("Processed", "0")
                    col_metrics[1].metric("Remaining", str(total_rows))
                    col_metrics[2].metric("Rate (rows/sec)", "0")
                    col_metrics[3].metric("ETA", "Calculating...")

                start_time = time.time()

                for i in range(0, total_rows, batch_size):
                    batch_end = min(i + batch_size, total_rows)
                    batch_df = df.iloc[i:batch_end]

                    batch_results = []
                    for idx, row in batch_df.iterrows():
                        try:
                            final_result, model_predictions = st.session_state.ensemble.extract_with_voting(row["text"])
                            result_row = {
                                "original_index": idx,
                                "text_preview": row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"],
                                **final_result,
                            }
                            if show_model_breakdown:
                                result_row["model_breakdown"] = json.dumps(model_predictions)
                            batch_results.append(result_row)
                        except Exception as e:
                            batch_results.append({
                                "original_index": idx,
                                "text_preview": row["text"][:100] + "...",
                                "error": str(e)
                            })

                    results.extend(batch_results)

                    # Progress update
                    progress = batch_end / total_rows
                    main_progress.progress(progress)

                    elapsed_time = time.time() - start_time
                    processing_rate = batch_end / elapsed_time if elapsed_time > 0 else 0
                    remaining_rows = total_rows - batch_end
                    eta_seconds = remaining_rows / processing_rate if processing_rate > 0 else 0

                    status_container.info(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}")

                    with metrics_container:
                        col_metrics = st.columns(4)
                        col_metrics[0].metric("Processed", f"{batch_end:,}")
                        col_metrics[1].metric("Remaining", f"{remaining_rows:,}")
                        col_metrics[2].metric("Rate (rows/sec)", f"{processing_rate:.1f}")
                        col_metrics[3].metric("ETA", f"{eta_seconds/60:.1f} min" if eta_seconds > 60 else f"{eta_seconds:.0f} sec")

                    if show_intermediate and results:
                        current_results_df = pd.DataFrame(results)
                        results_container.dataframe(current_results_df.tail(50), use_container_width=True)

                    time.sleep(0.1)

                st.session_state.results_df = pd.DataFrame(results)
                st.success(f"✅ Processing completed! Extracted {total_rows} rows in {elapsed_time:.1f}s")

# -------------------------
# Column 2 - Stats
# -------------------------
with col2:
    st.header("Real-time Stats")

    if st.session_state.results_df is not None:
        df_results = st.session_state.results_df
        st.metric("Total Rows Processed", len(df_results))

        if len(df_results) > 0:
            st.subheader("Field Extraction Success")
            for col in df_results.columns:
                if col not in ["original_index", "text_preview", "error", "model_breakdown"]:
                    non_null_count = df_results[col].notna().sum()
                    success_rate = (non_null_count / len(df_results)) * 100
                    st.metric(f"{col.title()} Success", f"{success_rate:.1f}%")

# =========================
# Download Results
# =========================
# Replace your download results section with this:

if st.session_state.results_df is not None:
    st.header("📥 Download Results")

    # Define the exact column mapping you want
    desired_columns = [
        'reporter_name', 'person_involved', 'incident_date', 'incident_time',
        'department', 'incident_description', 'location', 'label',
        'was_injured', 'injury_description'
    ]
    
    # Create mapping from your current columns to desired headers
    # You'll need to adjust the keys based on what your extractor actually produces
    column_mapping = {
        # Map your current column names to the desired ones
        # Example mappings (adjust these based on your actual column names):
        'name': 'reporter_name',
        'person': 'person_involved', 
        'date': 'incident_date',
        'time': 'incident_time',
        'dept': 'department',
        'description': 'incident_description',
        'loc': 'location',
        'category': 'label',
        'injured': 'was_injured',
        'injury': 'injury_description'
        # Add more mappings as needed
    }
    
    # Debug: Show current vs desired columns
    with st.expander("🔍 Column Mapping Debug", expanded=False):
        st.write("**Current columns in results:**")
        current_cols = [col for col in st.session_state.results_df.columns 
                       if col not in ["original_index", "text_preview", "error", "model_breakdown"]]
        st.write(current_cols)
        
        st.write("**Desired columns:**")
        st.write(desired_columns)
        
        st.write("**Current mapping:**")
        st.write(column_mapping)
    
    # Create the download DataFrame with proper column names
    download_df = pd.DataFrame()
    
    # Initialize all desired columns with empty values
    for col in desired_columns:
        download_df[col] = ""
    
    # Map existing data to the desired columns
    for old_col, new_col in column_mapping.items():
        if old_col in st.session_state.results_df.columns and new_col in desired_columns:
            download_df[new_col] = st.session_state.results_df[old_col].fillna("")
    
    # Fill any unmapped columns with data from results if column names match exactly
    for col in desired_columns:
        if col in st.session_state.results_df.columns and download_df[col].eq("").all():
            download_df[col] = st.session_state.results_df[col].fillna("")
    
    # Show preview
    st.subheader("Preview of Structured Download Data:")
    st.dataframe(download_df.head(), use_container_width=True)
    st.info(f"📊 Download will contain {len(download_df)} rows with standardized column headers")
    
    # Download options
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col_dl1:
        csv_buffer = io.StringIO()
        download_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Download Structured CSV",
            data=csv_buffer.getvalue(),
            file_name=f"incident_report_data_{timestamp}.csv",
            mime="text/csv",
            help="CSV with standardized column headers"
        )
    
    with col_dl2:
        # Excel version
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            download_df.to_excel(writer, index=False, sheet_name='Incident_Reports')
        st.download_button(
            label="📈 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=f"incident_report_data_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col_dl3:
        json_data = download_df.to_json(orient="records", indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"incident_report_data_{timestamp}.json",
            mime="application/json"
        )
    
    # Show extraction success rates
    st.subheader("📈 Field Extraction Success:")
    cols = st.columns(5)
    for i, col in enumerate(desired_columns):
        with cols[i % 5]:
            non_empty = download_df[col].ne("").sum()
            success_rate = (non_empty / len(download_df)) * 100 if len(download_df) > 0 else 0
            st.metric(f"{col.replace('_', ' ').title()}", f"{success_rate:.1f}%")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("Built by 2025 SWSR Team | Multi-Model Ensemble Entity Extraction")
