# sheets.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

def display_google_sheets_section(spreadsheet_name: str, worksheet_name: str = "Sheet2"):
    """
    Display a Google Sheets section in Streamlit and return a DataFrame.
    Credentials are loaded from st.secrets (configured in .streamlit/secrets.toml).
    Uses get_all_values() to avoid duplicate header issues.
    """
    st.markdown("---")
    st.header("📑 Import Data from Google Sheets")

    if st.button("Fetch Google Sheets Data", use_container_width=True):
        try:
            # Use broader scopes: spreadsheets + drive
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.readonly"
            ]

            # Load credentials from st.secrets
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=scopes,
            )
            gc = gspread.authorize(creds)

            # Debug: show which service account is being used
            st.write("🔑 Using service account:", st.secrets["gcp_service_account"]["client_email"])

            # Access sheet + worksheet
            sh = gc.open(spreadsheet_name)
            worksheet = sh.worksheet(worksheet_name)

            # Fetch raw values
            values = worksheet.get_all_values()

            if not values:
                st.warning("⚠️ The worksheet is empty.")
                return None

            # First row = headers (may contain duplicates/empties)
            raw_headers = values[0]

            # Deduplicate headers using Pandas internal helper
            parser = pd.io.parsers.ParserBase({'names': raw_headers})
            headers = parser._maybe_dedup_names(raw_headers)

            # Remaining rows = data
            df = pd.DataFrame(values[1:], columns=headers)

            st.success(f"✅ Fetched {len(df)} rows from **{spreadsheet_name}** ({worksheet_name})")
            st.dataframe(df.tail(5), use_container_width=True)

            return df

        except Exception as e:
            st.error(f"❌ Failed to fetch Google Sheets data: {e}")
            return None

    return None
