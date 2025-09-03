# sheets.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

def make_unique_headers(headers):
    """Ensure headers are unique and non-empty."""
    seen = {}
    newheaders = []
    for i, h in enumerate(headers):
        h = h.strip() if h else ""  # remove whitespace
        if h == "":
            h = f"col{i+1}"  # replace empty with col#
        if h in seen:
            seen[h] += 1
            h = f"{h}{seen[h]}"  # rename duplicate
        else:
            seen[h] = 0
        new_headers.append(h)
    return new_headers

def preprocessdataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard preprocessing to a DataFrame."""
    df = df.dropna(how="all")  # drop fully empty rows
    df = df.fillna("Missing")  # replace NaN
    df.columns = [
        c.strip().lower().replace(" ", "") for c in df.columns
    ]  # clean headers
    return df

def fetch_google_sheets(spreadsheet_name: str, worksheet_name: str = "Sheet2"):
    """Fetch data from a specific Google Sheets worksheet and return DataFrame."""
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ]

        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes,
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

        st.success(f"✅ Fetched {len(df)} rows from {spreadsheet_name} ({worksheet_name})")
        st.dataframe(df.head(5), use_container_width=True)

        return df

    except Exception as e:
        st.error(f"❌ Failed to fetch Google Sheets data: {e}")
        return None

def display_data_section(spreadsheet_name: str, worksheet_name: str = "Sheet2"):
    """
    Streamlit section to let users choose between Google Sheets (Sheet2) or file upload.
    Returns a preprocessed DataFrame.
    """
    st.markdown("---")
    st.header("📊 Import Data")

    option = st.radio(
        "Choose a data source:",
        ("Google Sheets (Sheet2)", "Upload a file"),
        horizontal=True,
    )

    df = None

    if option == "Google Sheets (Sheet2)":
        if st.button("Fetch Google Sheets Data", use_container_width=True):
            df = fetch_google_sheets(spreadsheet_name, worksheet_name)

    elif option == "Upload a file":
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Uploaded {uploaded_file.name}")
            st.dataframe(df.head(5), use_container_width=True)

    if df is not None:
        df = preprocess_dataframe(df)

        st.markdown("### 🛠️ Preprocessed Data")
        st.dataframe(df.head(10), use_container_width=True)

        return df

    return None

# 🔄 Backward compatibility wrapper so app.py keeps working
def display_google_sheets_section(spreadsheet_name: str, worksheet_name: str = "Sheet2"):
    """
    Wrapper to maintain compatibility with app.py.
    Uses the same logic as display_data_section but defaults to Sheet2.
    """
    return fetch_google_sheets(spreadsheet_name, worksheet_name)