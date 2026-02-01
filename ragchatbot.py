import streamlit as st

st.header("My First Streamlit App")

with st.sidebar:
    st.title("your Document ")
    file=st.file_uploader("Upload your PDF document here", type="pdf")