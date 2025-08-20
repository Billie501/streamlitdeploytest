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
    Loads the spaCy model. Downloads it if not available.
    More robust error handling for Streamlit Cloud.
    """
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except (OSError, ImportError) as e:
        st.info("🔄 Setting up spaCy model for first use...")
        
        try:
            # Try direct spacy download first
            import subprocess
            import sys
            
            with st.spinner("Downloading spaCy model 'en_core_web_sm'..."):
                result = subprocess.run([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    import spacy
                    return spacy.load("en_core_web_sm")
                else:
                    st.error(f"Failed to download spaCy model: {result.stderr}")
                    
        except Exception as download_error:
            st.error(f"Error downloading spaCy model: {str(download_error)}")
            
            # Fallback: suggest manual installation
            st.code("pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl")
            st.stop()
    
    except Exception as e:
        st.error(f"Unexpected error loading spaCy: {str(e)}")
        st.stop()

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

# =========================
# Header
# =========================
st.title("🤖 Real-time ML Entity Extraction Pipeline")
st.markdown("Multi-Model Ensemble with Voting for unstructured data classification")

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
                st.session_state.ensemble = EnsembleVotingExtractor(nlp=nlp)

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
if st.session_state.results_df is not None:
    st.header("📥 Download Results")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_buffer = io.StringIO()
        st.session_state.results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"extracted_data_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

    with col_dl2:
        json_data = st.session_state.results_df.to_json(orient="records", indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"extracted_data_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json"
        )

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Multi-Model Ensemble Entity Extraction")
