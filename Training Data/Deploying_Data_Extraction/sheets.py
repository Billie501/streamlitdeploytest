# sheets.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


def display_google_sheets_section(
    service_account_file: str,
    spreadsheet_name: str,
    worksheet_name: str = "Sheet1",
):
    """
    Display a Google Sheets section in Streamlit and return a DataFrame.

    Args:
        service_account_file (str): Path to your Google service account JSON file.
        spreadsheet_name (str): Name of the Google Spreadsheet.
        worksheet_name (str, optional): Worksheet/tab name. Defaults to "Sheet1".

    Returns:
        pd.DataFrame | None: DataFrame of sheet contents, or None if failed.
    """
    st.markdown("---")
    st.header("📑 Import Data from Google Sheets")

    if st.button("Fetch Google Sheets Data", use_container_width=True):
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
            creds = Credentials.from_service_account_file(service_account_file, scopes=scopes)
            gc = gspread.authorize(creds)

            sh = gc.open(spreadsheet_name)
            worksheet = sh.worksheet(worksheet_name)
            all_rows = worksheet.get_all_records()
            df = pd.DataFrame(all_rows)

            st.success(f"✅ Fetched {len(df)} rows from **{spreadsheet_name}** ({worksheet_name})")
            st.dataframe(df.tail(5), use_container_width=True)

            return df

        except Exception as e:
            st.error(f"❌ Failed to fetch Google Sheets data: {e}")
            return None

    return None
