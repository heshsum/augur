"""This is a module to use Streamlit and Facebook Prophet in order to predict growth."""

from io import StringIO
import time
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

DEFAULT_DAYS = 90

st.title('Augur - Forecasting for mailbox.org')

st.header('About')
st.markdown('This tool uses Facebook Prophet to predict growth based on the historic flight path.')

st.header('Usage')
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

    st.write('Calculated predictions:')
    with st.expander('Show raw data'):
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        st.table(forecast_data.sort_values(by='ds', ascending=False))
        csv = forecast_data.to_csv().encode('utf-8')
        st.download_button(
            label='Download as CSV',
            data=csv,
            file_name='augur.csv',
            mime='text/csv',
        )
    st.write(plot_plotly(model, forecast))


uploaded_file = st.file_uploader('Select CSV file')
period = st.number_input('Prediction period (days)', step=1, value=DEFAULT_DAYS)
# period = st.date_input(label='Enter prediction target',
#   value=datetime.date.today(), format='DD.MM.YYYY')

start_button = st.button('Start', type='primary')

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
        warning = st.warning('Please upload a file', icon='⚠️')
        time.sleep(3)
        warning.empty()
