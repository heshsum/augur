"""This is a module to use Streamlit and Facebook Prophet in order to predict growth."""

from io import StringIO
import time
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

DEFAULT_DAYS = 90

st.set_page_config(page_title='Augur forecasting', page_icon=None, layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title('Augur - Forecasting for mailbox.org')

st.subheader('About')
st.markdown('This tool estimates future growth based on the historic flight path.')

st.subheader('Usage')
st.markdown('''Upload a CSV file containing two columns:  
1. ds for datestamp (yyyy-mm-dd, e.g. 2023-12-31)
2. y for value (e.g. 42)''')


def predict(df):
    """Function to run the prediction for a given Dataframe object."""

    df.head()

    with st.spinner('Looking into the crystal ball...'):
        model = Prophet()
        model.fit(df)
        forecast = model.predict(model.make_future_dataframe(periods=period))

    with tab1:
        st.write(plot_plotly(model, forecast))
    with tab2:
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        csv = forecast_data.to_csv().encode('utf-8')
        st.download_button(
            label='Download as CSV',
            data=csv,
            file_name='augur.csv',
            mime='text/csv',
        )
        st.table(forecast_data.sort_values(by='ds', ascending=False))


uploaded_file = st.file_uploader('Select CSV file')
period = st.number_input('Prediction period (days)', step=1, value=DEFAULT_DAYS)

start_button = st.button('Start', type='primary')

tab1, tab2 = st.tabs(["Chart", "Raw data"])

if start_button:
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))

        # To read file as string:
        string_data = stringio.read()

        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        predict(df)
    else:
        # If there's no file, show a warning message for three seconds
        warning = st.warning('Please upload a file', icon='⚠️')
        time.sleep(3)
        warning.empty()
