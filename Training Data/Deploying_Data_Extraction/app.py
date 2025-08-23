# app.py
import streamlit as st
import pandas as pd
import time
import json
import io
from datetime import datetime
import spacy
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your custom extractor
from extractors import EnsembleVotingExtractor

# =========================
# Enhanced SpaCy Model Loader
# =========================
@st.cache_resource
def load_spacy_model():
    """
    Loads the spaCy model with comprehensive fallback and validation.
    """
    import spacy
    from spacy.lang.en import English
    
    try:
        # Try to load the full model first
        nlp = spacy.load("en_core_web_sm")
        st.success("✅ Loaded full spaCy model with NER capabilities")
        return nlp
    except OSError:
        try:
            # Try to load medium model
            nlp = spacy.load("en_core_web_md")
            st.success("✅ Loaded spaCy medium model")
            return nlp
        except OSError:
            pass
    
    # If no trained model available, create enhanced blank model
    st.warning("⚠️ No trained spaCy model found. Creating enhanced fallback model.")
    
    try:
        nlp = English()
        
        # Add essential pipeline components
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
            
        # Add basic entity ruler for common patterns
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
            
            # Add common patterns for fallback
            patterns = [
                {"label": "PERSON", "pattern": [{"TEXT": {"REGEX": r"^[A-Z][a-z]+ [A-Z][a-z]+$"}}]},
                {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]},
                {"label": "TIME", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}:\d{2}"}}]},
                {"label": "ORG", "pattern": [{"LOWER": {"IN": ["department", "warehouse", "office", "facility"]}}]},
            ]
            ruler.add_patterns(patterns)
        
        st.info("✅ Created enhanced fallback model with pattern matching")
        return nlp
        
    except Exception as e:
        st.error(f"Failed to create enhanced model: {str(e)}")
        return spacy.blank("en")

# =========================
# Enhanced Extraction Functions
# =========================
class ImprovedExtractor:
    def __init__(self):
        self.nlp = load_spacy_model()
        self.confidence_threshold = 0.3
        
    def extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """
        Enhanced pattern-based extraction with confidence scoring
        """
        results = {}
        
        # Date patterns
        date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        # Time patterns
        time_patterns = [
            r'\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b',
            r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b'
        ]
        
        # Name patterns (improved)
        name_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'(?:employee|worker|person|individual)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        
        # Department/Location patterns
        dept_patterns = [
            r'\b(warehouse|office|facility|department|production|maintenance|administration|security)\b',
            r'\b(floor\s+\d+|level\s+\d+|room\s+\d+|building\s+[A-Z])\b'
        ]
        
        # Injury patterns
        injury_patterns = [
            r'\b(cut|bruise|burn|sprain|fracture|injury|hurt|pain|ache|wound)\b',
            r'\b(injured|hurt|wounded|burned|cut|bruised)\b'
        ]
        
        text_lower = text.lower()
        
        # Extract dates
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['incident_date'] = matches[0]
                break
        
        # Extract times  
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['incident_time'] = matches[0]
                break
                
        # Extract names
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the first match as reporter, second as person involved
                if len(matches) >= 1:
                    results['reporter_name'] = matches[0]
                if len(matches) >= 2:
                    results['person_involved'] = matches[1]
                break
        
        # Extract department/location
        for pattern in dept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['department'] = matches[0]
                results['location'] = matches[0]
                break
        
        # Check for injuries
        injury_found = False
        for pattern in injury_patterns:
            if re.search(pattern, text_lower):
                injury_found = True
                break
        
        results['was_injured'] = 'Yes' if injury_found else 'No'
        
        # Extract injury description if injury detected
        if injury_found:
            # Look for sentences containing injury keywords
            sentences = text.split('.')
            for sentence in sentences:
                for pattern in injury_patterns:
                    if re.search(pattern, sentence.lower()):
                        results['injury_description'] = sentence.strip()
                        break
                if 'injury_description' in results:
                    break
        
        # Use entire text as incident description (truncated)
        results['incident_description'] = text[:500] + "..." if len(text) > 500 else text
        
        # Simple categorization
        if any(word in text_lower for word in ['slip', 'fall', 'trip']):
            results['label'] = 'Fall/Slip'
        elif any(word in text_lower for word in ['cut', 'sharp', 'blade']):
            results['label'] = 'Cut/Laceration'
        elif any(word in text_lower for word in ['burn', 'hot', 'fire']):
            results['label'] = 'Burn'
        elif any(word in text_lower for word in ['equipment', 'machine', 'malfunction']):
            results['label'] = 'Equipment'
        else:
            results['label'] = 'General'
        
        return results
    
    def combine_extractions(self, spacy_result: Dict, pattern_result: Dict, confidence_scores: Dict) -> Dict[str, Any]:
        """
        Intelligently combine results from different extraction methods
        """
        final_result = {}
        
        for field in ['reporter_name', 'person_involved', 'incident_date', 'incident_time',
                     'department', 'incident_description', 'location', 'label',
                     'was_injured', 'injury_description']:
            
            candidates = []
            
            # Add spaCy result if exists and confidence is good
            if field in spacy_result and spacy_result[field] and confidence_scores.get(f'spacy_{field}', 0) > self.confidence_threshold:
                candidates.append((spacy_result[field], confidence_scores[f'spacy_{field}']))
            
            # Add pattern result if exists
            if field in pattern_result and pattern_result[field]:
                # Pattern matching gets base confidence of 0.7
                candidates.append((pattern_result[field], 0.7))
            
            # Choose best candidate
            if candidates:
                # Sort by confidence and take the best
                best_candidate = max(candidates, key=lambda x: x[1])
                final_result[field] = best_candidate[0]
            else:
                final_result[field] = ""
        
        return final_result

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Enhanced ML Entity Extraction Pipeline",
    page_icon="🤖",
    layout="wide"
)

# =========================
# Custom CSS for better readability
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .success-metric {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Session State Init
# =========================
if "ensemble" not in st.session_state:
    st.session_state.ensemble = None
if "improved_extractor" not in st.session_state:
    st.session_state.improved_extractor = ImprovedExtractor()
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "results_df" not in st.session_state:
    st.session_state.results_df = None

# =========================
# Enhanced Diagnostics
# =========================
def enhanced_diagnostics():
    """Enhanced diagnostic section with actionable insights"""
    
    with st.expander("🔍 Model Performance Diagnostics", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SpaCy Model Status")
            nlp = load_spacy_model()
            
            st.write(f"**Model:** {nlp.meta.get('name', 'Custom/Blank')}")
            st.write(f"**Components:** {', '.join(nlp.pipe_names)}")
            
            # Test extraction
            test_text = "John Smith reported that Mary Johnson was injured at the warehouse on March 15, 2024 at 2:30 PM when she slipped and cut her hand."
            doc = nlp(test_text)
            
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            st.write(f"**Entities Found:** {len(entities)}")
            
            if entities:
                for ent_text, ent_label in entities:
                    st.write(f"  - {ent_text} ({ent_label})")
            else:
                st.warning("No entities detected - model may be limited")
        
        with col2:
            st.subheader("Pattern Extraction Test")
            improved_extractor = st.session_state.improved_extractor
            pattern_result = improved_extractor.extract_with_patterns(
                "John Smith reported that Mary Johnson was injured at the warehouse on March 15, 2024 at 2:30 PM when she slipped and cut her hand."
            )
            
            st.write("**Pattern Extraction Results:**")
            for field, value in pattern_result.items():
                if value:
                    st.write(f"  - **{field}:** {value}")
            
            # Calculate extraction success rate
            filled_fields = sum(1 for v in pattern_result.values() if v and str(v).strip())
            total_fields = len(pattern_result)
            success_rate = (filled_fields / total_fields) * 100
            
            if success_rate > 70:
                st.success(f"✅ Pattern extraction: {success_rate:.1f}% success")
            elif success_rate > 40:
                st.warning(f"⚠️ Pattern extraction: {success_rate:.1f}% success")
            else:
                st.error(f"❌ Pattern extraction: {success_rate:.1f}% success")

# =========================
# Header
# =========================
st.markdown('<h1 class="main-header">🤖 Enhanced ML Entity Extraction Pipeline</h1>', unsafe_allow_html=True)
st.markdown("**Multi-Model Ensemble with Pattern Fallback for Robust Entity Extraction**")

enhanced_diagnostics()

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ Configuration")
batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=500, value=100)
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
use_pattern_fallback = st.sidebar.checkbox("Enable Pattern Fallback", value=True, help="Use pattern matching when ML models fail")
show_intermediate = st.sidebar.checkbox("Show Intermediate Results", value=True)
show_model_breakdown = st.sidebar.checkbox("Show Model Breakdown", value=False)

# Update confidence threshold
st.session_state.improved_extractor.confidence_threshold = confidence_threshold

# =========================
# Main Layout
# =========================
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results & Analytics", "🔧 Advanced Settings"])

# -------------------------
# Tab 1 - Upload & Process
# -------------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your CSV file with unstructured incident reports",
            type=["csv"],
            help="CSV must contain a 'text' column with incident descriptions"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if "text" not in df.columns:
                    st.error("❌ CSV must contain a 'text' column")
                    st.stop()
                
                # Data validation
                valid_rows = df['text'].notna().sum()
                empty_rows = len(df) - valid_rows
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>📊 Data Summary</h4>
                    <ul>
                        <li><strong>Total Rows:</strong> {len(df):,}</li>
                        <li><strong>Valid Text Rows:</strong> {valid_rows:,}</li>
                        <li><strong>Empty Rows:</strong> {empty_rows:,}</li>
                        <li><strong>Columns:</strong> {', '.join(df.columns)}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Show sample data
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(3), use_container_width=True)
                
                # Quick text analysis
                if valid_rows > 0:
                    avg_length = df['text'].str.len().mean()
                    max_length = df['text'].str.len().max()
                    min_length = df['text'].str.len().min()
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    col_stat1.metric("Avg Text Length", f"{avg_length:.0f}")
                    col_stat2.metric("Max Text Length", f"{max_length:,}")
                    col_stat3.metric("Min Text Length", f"{min_length:,}")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.stop()
    
    with col2:
        st.header("🎯 Processing Options")
        
        # Processing method selection
        processing_method = st.radio(
            "Select Processing Method:",
            options=["Enhanced Pattern Matching", "Ensemble + Pattern Hybrid", "Ensemble Only"],
            index=0,
            help="Enhanced Pattern Matching is most reliable for new data"
        )
        
        if processing_method == "Ensemble + Pattern Hybrid" or processing_method == "Ensemble Only":
            st.warning("⚠️ Ensemble methods may require training and may not perform well on new data types")
    
    # Processing Section
    if uploaded_file is not None:
        st.header("🚀 Processing")
        
        if st.button("🎬 Start Processing", type="primary", use_container_width=True):
            
            results = []
            total_rows = len(df)
            
            # Create progress containers
            main_progress = st.progress(0)
            status_container = st.empty()
            
            # Metrics container
            metrics_container = st.container()
            with metrics_container:
                col_metrics = st.columns(5)
                processed_metric = col_metrics[0].empty()
                remaining_metric = col_metrics[1].empty()
                rate_metric = col_metrics[2].empty()
                eta_metric = col_metrics[3].empty()
                success_metric = col_metrics[4].empty()
            
            # Results preview container
            results_container = st.empty()
            
            start_time = time.time()
            successful_extractions = 0
            
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                batch_df = df.iloc[i:batch_end]
                
                batch_results = []
                batch_successes = 0
                
                for idx, row in batch_df.iterrows():
                    try:
                        text = str(row["text"]) if pd.notna(row["text"]) else ""
                        
                        if not text.strip():
                            result_row = {
                                "original_index": idx,
                                "text_preview": "Empty text",
                                "error": "Empty or invalid text"
                            }
                        else:
                            if processing_method == "Enhanced Pattern Matching":
                                # Use only pattern matching
                                final_result = st.session_state.improved_extractor.extract_with_patterns(text)
                                
                            elif processing_method == "Ensemble + Pattern Hybrid":
                                # Use ensemble with pattern fallback
                                if st.session_state.ensemble:
                                    try:
                                        ensemble_result, model_predictions = st.session_state.ensemble.extract_with_voting(text)
                                        pattern_result = st.session_state.improved_extractor.extract_with_patterns(text)
                                        
                                        # Combine results
                                        final_result = {}
                                        for field in pattern_result.keys():
                                            if field in ensemble_result and ensemble_result[field]:
                                                final_result[field] = ensemble_result[field]
                                            else:
                                                final_result[field] = pattern_result.get(field, "")
                                    except:
                                        final_result = st.session_state.improved_extractor.extract_with_patterns(text)
                                else:
                                    final_result = st.session_state.improved_extractor.extract_with_patterns(text)
                            
                            else:  # Ensemble Only
                                if st.session_state.ensemble:
                                    final_result, model_predictions = st.session_state.ensemble.extract_with_voting(text)
                                else:
                                    st.error("Ensemble not trained. Please train models first or use Pattern Matching.")
                                    st.stop()
                            
                            # Count successful extractions
                            filled_fields = sum(1 for v in final_result.values() if v and str(v).strip())
                            if filled_fields >= 3:  # At least 3 fields extracted
                                batch_successes += 1
                            
                            result_row = {
                                "original_index": idx,
                                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                                **final_result,
                                "extraction_success": filled_fields >= 3
                            }
                    
                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {str(e)}")
                        result_row = {
                            "original_index": idx,
                            "text_preview": text[:100] + "..." if 'text' in locals() else "Error loading text",
                            "error": str(e),
                            "extraction_success": False
                        }
                    
                    batch_results.append(result_row)
                
                results.extend(batch_results)
                successful_extractions += batch_successes
                
                # Update progress
                progress = batch_end / total_rows
                main_progress.progress(progress)
                
                elapsed_time = time.time() - start_time
                processing_rate = batch_end / elapsed_time if elapsed_time > 0 else 0
                remaining_rows = total_rows - batch_end
                eta_seconds = remaining_rows / processing_rate if processing_rate > 0 else 0
                
                # Update metrics
                processed_metric.metric("Processed", f"{batch_end:,}")
                remaining_metric.metric("Remaining", f"{remaining_rows:,}")
                rate_metric.metric("Rate (rows/sec)", f"{processing_rate:.1f}")
                eta_metric.metric("ETA", f"{eta_seconds/60:.1f} min" if eta_seconds > 60 else f"{eta_seconds:.0f} sec")
                success_metric.metric("Success Rate", f"{(successful_extractions/batch_end)*100:.1f}%")
                
                status_container.info(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}")
                
                # Show intermediate results
                if show_intermediate and results:
                    current_results_df = pd.DataFrame(results)
                    display_columns = [col for col in current_results_df.columns 
                                     if col not in ['original_index', 'error', 'extraction_success']]
                    results_container.dataframe(
                        current_results_df[display_columns].tail(10), 
                        use_container_width=True
                    )
                
                time.sleep(0.05)  # Small delay to prevent overwhelming
            
            # Final results
            st.session_state.results_df = pd.DataFrame(results)
            
            # Final summary
            total_time = time.time() - start_time
            final_success_rate = (successful_extractions / total_rows) * 100
            
            if final_success_rate > 70:
                st.success(f"✅ Processing completed successfully! {total_rows:,} rows processed in {total_time:.1f}s with {final_success_rate:.1f}% success rate")
            elif final_success_rate > 40:
                st.warning(f"⚠️ Processing completed with mixed results. {final_success_rate:.1f}% success rate")
            else:
                st.error(f"❌ Processing completed with low success rate: {final_success_rate:.1f}%")
                st.info("💡 Consider using 'Enhanced Pattern Matching' method or improving your data quality")

# -------------------------
# Tab 2 - Results & Analytics  
# -------------------------
with tab2:
    if st.session_state.results_df is not None:
        df_results = st.session_state.results_df
        
        st.header("📊 Extraction Results & Analytics")
        
        # Overall metrics
        col_summary = st.columns(4)
        total_processed = len(df_results)
        successful_extractions = df_results.get('extraction_success', pd.Series([False]*len(df_results))).sum()
        error_count = df_results['error'].notna().sum() if 'error' in df_results.columns else 0
        
        col_summary[0].metric("Total Processed", f"{total_processed:,}")
        col_summary[1].metric("Successful Extractions", f"{successful_extractions:,}")
        col_summary[2].metric("Errors", f"{error_count:,}")
        col_summary[3].metric("Overall Success Rate", f"{(successful_extractions/total_processed)*100:.1f}%")
        
        # Field-by-field analysis
        st.subheader("📈 Field Extraction Success Rates")
        
        desired_fields = ['reporter_name', 'person_involved', 'incident_date', 'incident_time',
                         'department', 'incident_description', 'location', 'label',
                         'was_injured', 'injury_description']
        
        # Create field success rate visualization
        field_stats = []
        for field in desired_fields:
            if field in df_results.columns:
                non_empty = df_results[field].notna() & (df_results[field] != "") & (df_results[field] != "None")
                success_count = non_empty.sum()
                success_rate = (success_count / len(df_results)) * 100
                field_stats.append({
                    'Field': field.replace('_', ' ').title(),
                    'Success_Count': success_count,
                    'Success_Rate': success_rate
                })
        
        if field_stats:
            stats_df = pd.DataFrame(field_stats)
            
            # Display in columns
            cols = st.columns(5)
            for i, row in stats_df.iterrows():
                col_idx = i % 5
                with cols[col_idx]:
                    success_rate = row['Success_Rate']
                    if success_rate > 70:
                        color = "success"
                    elif success_rate > 40:
                        color = "warning"  
                    else:
                        color = "error"
                    
                    st.markdown(f"""
                    <div class="{color}-metric">
                        <h4>{row['Field']}</h4>
                        <h2>{success_rate:.1f}%</h2>
                        <p>{row['Success_Count']:,} / {total_processed:,} rows</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Sample results preview
        st.subheader("📋 Sample Extraction Results")
        
        # Show successful extractions
        if successful_extractions > 0:
            success_df = df_results[df_results.get('extraction_success', False) == True]
            if len(success_df) > 0:
                display_cols = [col for col in success_df.columns 
                              if col not in ['original_index', 'error', 'extraction_success']]
                st.dataframe(success_df[display_cols].head(10), use_container_width=True)
        
        # Show errors if any
        if error_count > 0:
            st.subheader("❌ Processing Errors")
            error_df = df_results[df_results['error'].notna()]
            st.dataframe(error_df[['text_preview', 'error']].head(5), use_container_width=True)
        
        # Download section
        st.header("📥 Download Results")
        
        # Prepare download data with standardized columns
        download_df = pd.DataFrame()
        
        # Initialize all desired columns
        for col in desired_fields:
            download_df[col] = ""
        
        # Fill with available data
        for col in desired_fields:
            if col in df_results.columns:
                download_df[col] = df_results[col].fillna("")
        
        # Download options
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with col_dl1:
            csv_buffer = io.StringIO()
            download_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📊 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"incident_extraction_{timestamp}.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            json_data = download_df.to_json(orient="records", indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_data,
                file_name=f"incident_extraction_{timestamp}.json",
                mime="application/json"
            )
        
        with col_dl3:
            # Excel with formatting
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                download_df.to_excel(writer, index=False, sheet_name='Extractions')
            st.download_button(
                label="📈 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"incident_extraction_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("👆 Upload and process data first to see results and analytics")

# -------------------------  
# Tab