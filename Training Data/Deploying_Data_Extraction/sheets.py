# sheets.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


def make_unique_headers(headers):
    """Ensure headers are unique and non-empty."""
    seen = {}
    new_headers = []
    for i, h in enumerate(headers):
        h = h.strip() if h else ""  # remove whitespace
        if h == "":
            h = f"col_{i+1}"  # replace empty with col_#
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"  # rename duplicate
        else:
            seen[h] = 0
        new_headers.append(h)
    return new_headers


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard preprocessing to a DataFrame."""
    df = df.dropna(how="all")  # drop fully empty rows
    df = df.fillna("Missing")  # replace NaN
    df.columns = [
        c.strip().lower().replace(" ", "_") for c in df.columns
    ]  # clean headers
    return df


def fetch_google_sheets(spreadsheet_name: str, worksheet_name: str = "Sheet2") -> pd.DataFrame | None:
    """Fetch data from a specific Google Sheets worksheet and return DataFrame."""
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        gc = gspread.authorize(creds)

        st.write("🔑 Using service account:", st.secrets["gcp_service_account"]["client_email"])

        sh = gc.open(spreadsheet_name)
        worksheet = sh.worksheet(worksheet_name)

        values = worksheet.get_all_values()
        if not values:
            st.warning("⚠️ The worksheet is empty.")
            return None

        raw_headers = values[0]
        headers = make_unique_headers(raw_headers)

        df = pd.DataFrame(values[1:], columns=headers)

        st.success(f"✅ Fetched {len(df)} rows from **{spreadsheet_name}** ({worksheet_name})")
        st.dataframe(df.head(5), use_container_width=True)
        return df

    except Exception as e:
        st.error(f"❌ Failed to fetch Google Sheets data: {e}")
        return None


def display_data_section(spreadsheet_name: str, worksheet_name: str = "Sheet2") -> pd.DataFrame | None:
    """
    Streamlit UI to select data source (Google Sheets or upload) with preprocessing toggle.
    Returns a DataFrame according to user choice.
    """
    st.markdown("---")
    st.header("📊 Import Data")

    # Step 1: Choose data source
    source_option = st.radio(
        "Choose a data source:",
        ("Google Sheets (Sheet2)", "Upload a file"),
        horizontal=True,
    )

    df = None

    # Step 2: Checkbox for preprocessing
    apply_preprocessing = st.checkbox("🛠️ Apply preprocessing", value=True, help="Toggle preprocessing for raw data")

    # Step 3: Fetch or upload
    if source_option == "Google Sheets (Sheet2)":
        if st.button("Fetch Google Sheets Data", use_container_width=True):
            df = fetch_google_sheets(spreadsheet_name, worksheet_name)

    elif source_option == "Upload a file":
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Uploaded {uploaded_file.name}")
            st.dataframe(df.head(5), use_container_width=True)

    # Step 4: Apply preprocessing if selected
    if df is not None:
        if apply_preprocessing:
            df = preprocess_dataframe(df)
            st.markdown("### 🛠️ Preprocessed Data")
        else:
            st.markdown("### 📄 Raw Data")

        st.dataframe(df.head(10), use_container_width=True)
        return df

    return None


# Backward compatibility for your existing app.py
def display_google_sheets_section(spreadsheet_name: str, worksheet_name: str = "Sheet2") -> pd.DataFrame | None:
    """
    Wrapper to maintain backward compatibility.
    Returns Google Sheets data without preprocessing.
    """
    return fetch_google_sheets(spreadsheet_name, worksheet_name)
