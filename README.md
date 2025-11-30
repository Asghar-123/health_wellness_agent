# Health Agent

This project implements a conversational AI health agent designed to assist users with health-related queries, meal planning, workout recommendations, and goal tracking. The agent leverages various specialized tools and guardrails to provide safe, accurate, and personalized interactions.

## Features

*   **Goal Analyzer:** Helps users set and track health goals.
*   **Meal Planner:** Generates personalized meal plans based on user preferences and dietary needs.
*   **Scheduler:** Assists with scheduling health-related activities and appointments.
*   **Tracker:** Allows users to log and monitor various health metrics.
*   **Workout Recommender:** Provides tailored workout routines and suggestions.
*   **Guardrails:** Incorporates safety measures and content filtering to ensure responsible AI interactions.

## Project Structure

```
.
├── agent.py               # Main agent orchestration logic
├── app.py                 # Application entry point (e.g., FastAPI, Flask) It is based on Streamlit now.
├── context.py             # Manages conversation context and session data
├── hooks.py               # Custom hooks or middleware
├── main.py                # Primary execution file
├── pyproject.toml         # Project configuration for poetry/uv
├── requirements.txt       # Python dependencies
├── guardrails/            # Contains modules for AI safety and content filtering
│   ├── disclaimer_generator.py # Generates disclaimers for health advice
│   ├── guardrail_manager.py    # Manages the application of guardrails
│   └── query_filters.py        # Filters and validates user queries
└── tools/                 # Specialized tools used by the health agent
    ├── goal_analyzer.py
    ├── meal_planner.py
    ├── scheduler.py
    ├── tracker.py
    └── workout_recommender.py
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/health_agent.git
    cd health_agent
    ```

2.  **Set up a virtual environment and install dependencies:**
    If you are using `uv`:
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```
    Alternatively, using `pip`:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root directory and add any necessary API keys or configuration settings. For example:
    ```
    OPENAI_API_KEY=your_openai_api_key or GEMINI_API_KEY=your api key(Currently it uses Gemini API)
    # Add other environment variables as needed
    ```

## Usage

To run the health agent application:

```bash
python main.py
```
(Modify this command if your application uses a different entry point, e.g., `uvicorn app:app --reload` for FastAPI)

Once running, interact with the agent through its exposed interface (e.g., a web UI or API endpoint).

## Testing

To run the tests:

```bash
pytest
```
