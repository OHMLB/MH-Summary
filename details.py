import streamlit as st  

def show_details():  
    st.title("MH Deviation & Schedule Delay")  
    st.write("Hello WTF")  
    if st.button("Back"):  
        st.session_state.page = "main"