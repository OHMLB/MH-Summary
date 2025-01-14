import streamlit as st  
from main import main
from details import show_details


# Initialize session state  
if "page" not in st.session_state:  
    st.session_state.page = "main"  

# Page routing  
if st.session_state.page == "main":  
    main()
elif st.session_state.page == "details":  
    show_details()