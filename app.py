# frontend/app.py
import streamlit as st
import requests
import json

# --- Configuration (can be moved to a separate config.py) ---
# Assuming your FastAPI backend runs locally on port 8000
# You can put this in frontend/config.py
FASTAPI_BACKEND_URL = "http://localhost:8000" 

# --- Session State Initialization ---
# Initialize chat history if it's not already in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add initial bot greeting
    st.session_state.chat_history.append({"role": "assistant", "content": "Hello! I can help you with patient medical records. What information are you looking for?"})

# Initialize state for current query's pagination
if "current_query_details" not in st.session_state:
    st.session_state.current_query_details = {
        "query": None,        # Stores the last user prompt that initiated a search
        "offset": 0,          # Current offset for fetching more results
        "has_more": False,    # True if there are more results to fetch
        "total_records": 0,   # Total unique records found for the last query
        "total_unique_patients": 0 # Total unique patients found for the last query
    }

# --- Constants for Pagination ---
INITIAL_DISPLAY_LIMIT = 5       # How many results to show initially
SUBSEQUENT_DISPLAY_LIMIT = 10   # How many results to show on "Show More" click

# --- Helper Functions for API Calls ---

def fetch_results_from_backend(prompt, offset, limit):
    """
    Makes a POST request to the FastAPI backend's /chat endpoint.
    """
    try:
        data = {
            "user_prompt": prompt,
            "offset": offset,
            "limit": limit
        }
        response = requests.post(
            f"{FASTAPI_BACKEND_URL}/chat",
            data=data,
            timeout=30 # Add a timeout for robustness
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to the backend at {FASTAPI_BACKEND_URL}. Please ensure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("The request to the backend timed out. The server might be busy or the query is taking too long.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data from the backend: {e}")
        try:
            error_details = response.json()
            st.error(f"Backend error details: {error_details.get('detail', error_details)}")
        except json.JSONDecodeError:
            st.error(f"Backend returned non-JSON response: {response.text}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding JSON response from backend. The backend might be returning invalid data.")
        return None

def handle_initial_message_send(user_prompt):
    """
    Handles sending a new message and fetching the initial batch of results.
    This function is called when the user submits a new prompt.
    """
    if user_prompt:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Reset pagination state for the new query
        st.session_state.current_query_details["query"] = user_prompt
        st.session_state.current_query_details["offset"] = 0
        st.session_state.current_query_details["has_more"] = False
        st.session_state.current_query_details["total_records"] = 0
        st.session_state.current_query_details["total_unique_patients"] = 0


        # Fetch initial results from the backend
        data = fetch_results_from_backend(
            st.session_state.current_query_details["query"],
            st.session_state.current_query_details["offset"],
            INITIAL_DISPLAY_LIMIT
        )

        if data:
            # Append bot messages to chat history
            for msg in data.get("messages", []):
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
            
            # Update pagination state
            st.session_state.current_query_details["offset"] = data.get("next_offset", 0)
            st.session_state.current_query_details["has_more"] = data.get("has_more", False)
            st.session_state.current_query_details["total_records"] = data.get("total_records", 0)
            st.session_state.current_query_details["total_unique_patients"] = data.get("total_unique_patients", 0)
            
            # Add final summary message if no more records
            if not data.get("has_more", False) and data.get("messages"):
                total_recs = data.get('total_records', 0)
                total_patients = data.get('total_unique_patients', 0)
                summary_msg = f"That's all unique records found for your request ({total_recs} record(s) related to {total_patients} unique patient(s))."
                st.session_state.chat_history.append({"role": "assistant", "content": summary_msg})
            elif not data.get("messages") and st.session_state.current_query_details["offset"] == 0:
                # If no messages and it was the initial query (offset 0),
                # the backend already sends a "couldn't find any info" message.
                # We append it here if it's the only message.
                if not data.get("messages"):
                    st.session_state.chat_history.append({"role": "assistant", "content": "I couldn't find any information matching your request. Could you please rephrase or provide more details?"})
        else:
            # If fetch_results_from_backend returned None due to an error,
            # an error message has already been displayed by fetch_results_from_backend.
            # We just add a generic fallback if needed.
            if not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != "assistant" or "error" not in st.session_state.chat_history[-1]["content"].lower():
                st.session_state.chat_history.append({"role": "assistant", "content": "Oops! Something went wrong while processing your request. Please try again."})


def handle_show_more():
    """
    Handles fetching more results when the "Show More" button is clicked.
    """
    current_query = st.session_state.current_query_details["query"]
    current_offset = st.session_state.current_query_details["offset"]

    if current_query:
        data = fetch_results_from_backend(current_query, current_offset, SUBSEQUENT_DISPLAY_LIMIT)
        if data:
            for msg in data.get("messages", []):
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
            
            st.session_state.current_query_details["offset"] = data.get("next_offset", current_offset)
            st.session_state.current_query_details["has_more"] = data.get("has_more", False)
            
            # If no more records after this fetch, add the final summary
            if not data.get("has_more", False) and data.get("messages"):
                total_recs = st.session_state.current_query_details['total_records']
                total_patients = st.session_state.current_query_details['total_unique_patients']
                summary_msg = f"That's all unique records found in this batch, and all available records for your request ({total_recs} record(s) related to {total_patients} unique patient(s))."
                st.session_state.chat_history.append({"role": "assistant", "content": summary_msg})
        else:
            # Error message already handled by fetch_results_from_backend
            pass # No additional message needed here


# --- Streamlit App Layout ---
st.set_page_config(page_title="Patient Medical Records Chatbot", layout="centered", initial_sidebar_state="collapsed")

st.title("🧑‍⚕️ Patient Medical Records Chatbot")

# Display all messages from the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input at the bottom of the chat interface
# The on_submit callback triggers a script rerun and executes handle_initial_message_send
user_prompt = st.chat_input(
    "Type your message...",
    on_submit=lambda: handle_initial_message_send(st.session_state.user_input),
    key="user_input" # Unique key for the input widget
)

# "Show More" button logic
# This button is only displayed if 'has_more' is True for the current query
if st.session_state.current_query_details["has_more"]:
    # Using st.button within a container helps with layout consistency
    st.button("Show More Results", on_click=handle_show_more, use_container_width=True)

# Note: Streamlit automatically handles scrolling to the bottom of the chat.