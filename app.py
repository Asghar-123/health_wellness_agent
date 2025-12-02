import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import json
import uuid
from agents import OpenAIChatCompletionsModel
from agent import HealthPlannerAgent
from context import UserSessionContext

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()
api_key = st.secrets["GEMINI_API_KEY"]

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def save_session_context(session_id: str, ctx: UserSessionContext):
    """Saves the UserSessionContext to a JSON file."""
    file_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    with open(file_path, "w") as f:
        json.dump(ctx.model_dump(), f, indent=4)

def load_session_context(session_id: str) -> UserSessionContext:
    """Loads the UserSessionContext from a JSON file, or creates a new one if not found."""
    file_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return UserSessionContext(**data)
        except (json.JSONDecodeError, TypeError):
            # If file is corrupted or not a valid context, create a new one
            return UserSessionContext(uid=session_id)
    return UserSessionContext(uid=session_id) # Create new if not found, use existing session_id

# Set Streamlit page configuration
st.set_page_config(page_title="Health & Wellness Assistant", page_icon="🏋️‍♀️", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #4169E1; /* Royal Blue */
        color: #000000 ; /* Pure white for main text */
    }

    /* General text elements */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText {
        color:#000000 ;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #3650B0; /* Darker shade of Royal Blue for sidebar */
        color:#000000 ; /* Ensure sidebar text is also white */
    }

    /* Chat bubble styling */
    [data-testid="chat-message-container"] {
        background-color: #3650B0; /* Lighter shade for chat bubbles */
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color:#000000 ; /* Ensure chat bubble text is white */
    }

    /* User chat bubble */
    [data-testid="chat-message-container"]:has(div[data-testid="chat-avatar-user"]) {
        background-color: #2F4FBB; /* A darker, more distinct blue for user messages */
    }

    /* Assistant chat bubble */
    [data-testid="chat-message-container"]:has(div[data-testid="chat-avatar-assistant"]) {
        background-color: #4C7AFB; /* A slightly lighter blue for assistant messages */
    }

    /* Button styling */
    .stButton>button {
        background-color: #3650B0; /* Same as sidebar background */
        color:#000000 ;
        border: 1px solid #FFFFFF; /* Add a light border to make it visible */
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #2F4FBB; /* Darker on hover */
        border-color: #FFFFFF;
    }
    .stButton>button:active {
        background-color: #263E9B; /* Even darker on click */
    }
    
    /* Expander styling */
    .st-emotion-cache-1h6xps {
        background-color: #5580F7; /* Same as chat bubbles */
        border-radius: 10px;
        color:#000000 ; /* Ensure expander text is white */
    }

    /* Adjust color for any remaining text to be white if needed */
    .st-emotion-cache-nahz7x, .st-emotion-cache-1avcm0n { /* Specific Streamlit classes for text */
        color:#000000 ;
    }

</style>
""", unsafe_allow_html=True)


# Initialize user session context
if "user_context" not in st.session_state:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    st.session_state.user_context = load_session_context(st.session_state.session_id)

# Sidebar
with st.sidebar:
    st.header("Navigation")

    if st.button("Clear Chat History"):
        # Clear the chat history
        st.session_state.user_context.chat_history = []
        # Save the cleared context to make it persistent
        save_session_context(st.session_state.session_id, st.session_state.user_context)
        # Rerun the app to update the UI
        st.rerun()

    # Description of Agent Dropdown
    st.subheader("Agent Description")
    with st.expander("Learn about this assistant"):
        st.write("**Purpose:** To provide general health, nutrition, workout, and biology information.")
        st.write("**Topics:** Meal plans, workout routines, goal tracking, injury support, nutrition expert advice, and general health inquiries.")
        st.write("**Goals:** To assist users in achieving their health and wellness objectives by offering personalized (where applicable) and informative guidance.")

    st.markdown("---")

    # About Section
    st.subheader("About")
    with st.expander("Learn more about this agent"):
        st.write("This Health & Wellness Assistant is designed to provide general health, nutrition, workout, and biology information. It can help with meal plans, workout routines, and goal tracking.")
        st.write("Developed by Syed Muhammad Asghar Ali Rizvi.")


# Initialize HealthPlannerAgent if not already in session state
@st.cache_resource
def get_health_agent():
    return HealthPlannerAgent()

if "health_agent" not in st.session_state:
    st.session_state.health_agent = get_health_agent()

# Initialize chat history in context if it doesn't exist
if not hasattr(st.session_state.user_context, "chat_history"):
    st.session_state.user_context.chat_history = []


st.title("🏋️‍♀️ Health & Wellness Assistant")
st.write("I'm here to provide general health, nutrition, workout, and biology information. I can help with meal plans, workout routines, and goal tracking.")
st.warning("🚨 **Disclaimer:** I am an AI assistant and cannot provide medical diagnoses, treatment advice, or emergency services. Always consult a qualified healthcare professional for medical concerns. In case of an emergency, please contact emergency services immediately.")

# Display chat history from the context
for message in st.session_state.user_context.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_query := st.chat_input("How can I help you today?"):
    # Add user message to chat history in context
    st.session_state.user_context.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Your agent is thinking..."):
            # Run the agent with the user query and the full context
            response_dict = asyncio.run(st.session_state.health_agent.run(user_query, st.session_state.user_context))
            agent_response = response_dict.get("response", "An error occurred or no specific response was generated.")

            # Display agent's response
            st.markdown(agent_response)
            
            # Add agent response to chat history in context
            st.session_state.user_context.chat_history.append({"role": "assistant", "content": agent_response})
            
            # Save the updated session context after each interaction
            save_session_context(st.session_state.session_id, st.session_state.user_context)

        # Optionally display structured data if available



        # Display handoff logs if any
        if hasattr(st.session_state.user_context, "handoff_logs") and st.session_state.user_context.handoff_logs:
            st.write("---")
            st.subheader("Handoff Logs:")
            for log in st.session_state.user_context.handoff_logs:
                st.write(log)
            # Clear logs after displaying to avoid repetition unless specifically desired
            st.session_state.user_context.handoff_logs = []
