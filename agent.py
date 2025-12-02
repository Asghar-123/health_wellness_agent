from __future__ import annotations
import os
import enum
import re
from typing import Any, Dict, List, Optional
import streamlit as st
from context import UserSessionContext
from hooks import RunHooks
from guardrails.guardrail_manager import GuardrailManager

from tools.goal_analyzer import GoalAnalyzerTool
from tools.meal_planner import MealPlannerTool
from tools.workout_recommender import WorkoutRecommenderTool
from tools.scheduler import CheckinSchedulerTool
from tools.tracker import ProgressTrackerTool

from agents import AsyncOpenAI, OpenAIChatCompletionsModel,Runner
from agents.run import RunConfig, ModelSettings
import typing_extensions
from dotenv import load_dotenv
load_dotenv()


class ModelTracing(enum.Enum):
    DISABLED = 0
    ENABLED = 1
    ENABLED_WITHOUT_DATA = 2

    def is_disabled(self) -> bool:
        return self == ModelTracing.DISABLED

    def include_data(self) -> bool:
        return self == ModelTracing.ENABLED


class HealthPlannerAgent:
    """
    A health planner agent that generates meal and workout plans based on user goals.
    Handles injuries, nutrition, and escalation logic internally.
    """

    def __init__(self):
        self._initialize_tools()
        self._initialize_model()
        self.guardrail_manager = GuardrailManager()
        self.hooks = RunHooks()

    def _initialize_tools(self):
        """Initializes the tools for the agent."""
        self.tools = {
            "goal_analyzer": GoalAnalyzerTool(),
            "meal_planner": MealPlannerTool(),
            "workout_recommender": WorkoutRecommenderTool(),
            "scheduler": CheckinSchedulerTool(),
            "tracker": ProgressTrackerTool(),
        }

    def _initialize_model(self):
        """Initializes the Gemini model."""
        gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print("ERROR: GEMINI_API_KEY is not set in environment variables.")
            raise ValueError("GEMINI_API_KEY is not set.")

        external_client = AsyncOpenAI(
            
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        self.model = OpenAIChatCompletionsModel(
            model="models/gemini-2.0-flash",
            openai_client=external_client,
        )

        self.config = RunConfig(
            model=self.model,
            model_provider=external_client,
            tracing_disabled=True,
        )

    async def _get_user_intent(self, user_input: str, ctx: UserSessionContext) -> str:
        """Classifies the user's intent based on their query."""
        system_prompt = """
You are an expert at classifying user intent. Your task is to classify the user's query into ONLY one of the following categories:

- "set_or_update_goal": The user is explicitly stating a goal or a desire to change their goal.
  Examples: "my goal is to lose weight", "i want to get fit", "change my goal to build muscle", "i want to increase my biceps".

- "ask_meal_plan": The user is specifically asking for a diet or meal plan.
  Examples: "give me a meal plan", "what should I eat for a week?", "suggest a diet plan".

- "ask_workout_plan": The user is specifically asking for a workout plan or exercises.
  Examples: "suggest some exercises", "can you give me a workout routine?", "what are some good bicep exercises?".

- "ask_general_question": The user is asking a general "what", "why", or "how" question, or is making a general statement that requires a conversational response. This is the default for most questions that are not a direct request for a plan or goal setting.
  Examples: "why is exercise important?", "what is a calorie?", "tell me more about that", "what can you do?".

- "log_water": The user wants to log their water intake.
  Examples: "i drank 500ml of water", "log my water".

- "handle_injury": The user is mentioning an injury, pain, or asking for medical advice.
  Examples: "i hurt my knee", "my back is in pain".

- "other": If the query does not fit any other category.
Answer should not exceed from this range 50 till 400 words 
If the user ask for meal plan or diet plan,you should provide the meal plan but first ask for their dietary preferences.

Respond ONLY with the single category name.
"""
        messages = ctx.chat_history + [{"role": "user", "content": user_input}]
        response_obj = await self.model.get_response(
            system_instructions=system_prompt,
            input=messages,
            model_settings=ModelSettings(temperature=0.0),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=getattr(ctx, 'previous_response_id', None),
        )
        if response_obj and getattr(response_obj, 'output', None):
            content = getattr(response_obj.output[0], 'content', None)
            if content and hasattr(content[0], 'text'):
                return content[0].text.strip()
        return "other" # Default intent if classification fails

    async def run(self, user_input: str, ctx: UserSessionContext) -> Dict[str, Any]:
        """Main entry point for handling user queries."""
        await self.hooks.on_agent_start("HealthPlannerAgent", ctx)

        # --- Guardrail Pre-processing ---
        passed_guardrail, refusal_message = self.guardrail_manager.pre_process_query(user_input)
        if not passed_guardrail:
            return {"ok": False, "response": refusal_message}

        intent = await self._get_user_intent(user_input, ctx)
        dynamic_instructions_tone = self._get_dynamic_instructions_tone(user_input)

        if intent == "handle_injury":
            return await self._handle_medical_query(user_input, ctx, dynamic_instructions=dynamic_instructions_tone)
        
        elif intent == "log_water":
            return self._process_water_intake(user_input)

        elif intent == "set_or_update_goal":
            ga = self.tools["goal_analyzer"]
            await self.hooks.on_tool_start(ga.name, user_input)
            parsed = await ga.run(self.model, user_input)
            if parsed.get("ok"):
                ctx.goal = parsed["goal"]
                plans_response = await self._generate_plans_and_response(ctx, dynamic_instructions_tone)
                if plans_response["ok"]:
                    plans_response["response"] = self.guardrail_manager.post_process_response(plans_response["response"])
                return plans_response
            else:
                return await self._process_general_query(
                    user_input, ctx,
                    system_instructions=dynamic_instructions_tone + " User query couldn't be parsed by goal analyzer."
                )
        
        elif intent == "ask_workout_plan":
            if not ctx.goal:
                # If no goal is set, try to parse one from the current query
                ga = self.tools["goal_analyzer"]
                parsed = await ga.run(self.model, user_input)
                if parsed.get("ok"):
                    ctx.goal = parsed["goal"]
                else:
                    # If goal can't be parsed, ask for it
                    response_text = "To suggest exercises, I need to know your fitness goal. What is your primary goal?"
                    return {"ok": True, "response": self.guardrail_manager.post_process_response(response_text)}
            
            # Now that a goal is set (or was already set), generate the workout plan
            workout_response = await self._generate_workout_plan(ctx, dynamic_instructions_tone)
            workout_response["response"] = self.guardrail_manager.post_process_response(workout_response["response"])
            return workout_response

       
        
        elif intent == "ask_general_question":
            return await self._process_general_query(user_input, ctx, system_instructions=dynamic_instructions_tone)

        else: # Fallback for "other" or failed intent classification
            system_instructions = (
                "You are a specialized AI assistant with expertise in health, biology, and medical queries, but if the user asked about diet or meal plan you should answer "
                "Provide accurate, safe, and helpful information. If the user's query is outside the scope, "
                "state that you are a specialized health and wellness assistant."
            )
            return await self._process_general_query(user_input, ctx, system_instructions)

    async def _handle_medical_query(self, user_input: str, ctx: UserSessionContext, dynamic_instructions: str) -> Dict[str, Any]:
        """Handles general medical-related queries with disclaimer."""
        medical_instruction = (
            f"You are a health information assistant, not a medical professional. "
            f"Provide general information and append the disclaimer: \"{self.guardrail_manager.medical_disclaimer_text}\"."
        )
        messages = ctx.chat_history + [{"role": "user", "content": user_input}]
        response_obj = await self.model.get_response(
            system_instructions=dynamic_instructions + " " + medical_instruction,
            input=messages,
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=getattr(ctx, 'previous_response_id', None),
        )
        ctx.previous_response_id = getattr(response_obj, 'response_id', None)
        response_text = "I'm sorry, I couldn't provide medical information at this time."
        if response_obj and getattr(response_obj, 'output', None):
            content = getattr(response_obj.output[0], 'content', None)
            if content and hasattr(content[0], 'text'):
                response_text = content[0].text
        return {"ok": True, "response": self.guardrail_manager.post_process_response(response_text)}

    async def _process_general_query(self, user_input: str, ctx: UserSessionContext, system_instructions: str) -> Dict[str, Any]:
        """Handles general queries using the model."""
        messages = ctx.chat_history + [{"role": "user", "content": user_input}]
        response_obj = await self.model.get_response(
            system_instructions=system_instructions,
            input=messages,
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=getattr(ctx, 'previous_response_id', None),
        )
        ctx.previous_response_id = getattr(response_obj, 'response_id', None)
        response_text = "I'm sorry, I couldn't generate a response at this time."
        if response_obj and getattr(response_obj, 'output', None):
            content = getattr(response_obj.output[0], 'content', None)
            if content and hasattr(content[0], 'text'):
                response_text = content[0].text
        return {"ok": True, "response": self.guardrail_manager.post_process_response(response_text)}

    def _process_water_intake(self, user_input: str) -> Dict[str, Any]:
        """Parses water intake amount from the user input."""
        amount_ml = 0
        match = re.search(r'(\d+)\s*(ml|milliliters|cup|cups|oz|ounces)', user_input.lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit in ["cup", "cups"]:
                amount_ml = amount * 240
            elif unit in ["oz", "ounces"]:
                amount_ml = amount * 30
            else:
                amount_ml = amount
            response_text = f"Logged {amount_ml}ml of water. (Hydration tracker functionality is currently simulated.)"
            return {"ok": True, "response": self.guardrail_manager.post_process_response(response_text)}
        else:
            response_text = "Please specify the amount of water to log (e.g., 'log 500ml water')."
            return {"ok": False, "response": self.guardrail_manager.post_process_response(response_text)}

    async def _parse_and_set_diet_preferences(self, user_input: str, ctx: UserSessionContext) -> bool:
        """Parses dietary preferences from user input and sets them in context."""
        system_instructions = (
            "You are a helpful assistant specialized in identifying diet preferences. "
            "Extract specific dietary restrictions, preferences, and food likes/dislikes. "
            "Respond ONLY with a concise comma-separated string. If none, respond with 'None'."
        )
        messages = ctx.chat_history + [{"role": "user", "content": user_input}]
        response_obj = await self.model.get_response(
            system_instructions=system_instructions,
            input=messages,
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=getattr(ctx, 'previous_response_id', None),
        )
        if response_obj and getattr(response_obj, 'output', None):
            content = getattr(response_obj.output[0], 'content', None)
            if content and hasattr(content[0], 'text'):
                parsed_preferences = content[0].text.strip()
                if parsed_preferences.lower() != "none":
                    ctx.diet_preferences = parsed_preferences
                    return True
        return False

    def _get_dynamic_instructions_tone(self, user_input: str) -> str:
        """Generates dynamic system instructions for model queries."""
        base_instruction = (
            "You are a specialized AI assistant with expertise in health, biology, and medical queries. "
            "Your primary goal is to provide accurate, safe, and helpful information. "
            "Purpose: To provide general health, nutrition, workout, and biology information. "
            "Topics: Meal plans, workout routines, goal tracking, injury support, nutrition expert advice, "
            "and general health inquiries, as well as personalized meal plans based on dietary preferences. "
            "Goals: To assist users in achieving their health and wellness objectives by offering personalized "
            "(where applicable) and informative guidance."
        )
        return base_instruction

    async def _generate_workout_plan(self, ctx: UserSessionContext, dynamic_instructions: str) -> Dict[str, Any]:
        """Generates a workout plan based on the user's goal."""
        response_parts = []
        goal = ctx.goal
        goal_name = goal.get('name', 'an unspecified goal')
        response_parts.append(f"Based on your goal to '{goal_name}', here is a suggested workout plan:")

        wr = self.tools["workout_recommender"]
        await self.hooks.on_tool_start(wr.name, {"goal": ctx.goal})
        workout = await wr.run(self.model, "beginner", ctx.goal)
        if workout and workout.get("workout_plan"):
            ctx.workout_plan = workout["workout_plan"]
            workout_plan_str = ["\nHere is your 7-day workout plan:"]
            for item in ctx.workout_plan:
                workout_plan_str.append(f"- {item}")
            response_parts.append("\n".join(workout_plan_str))
        
        final_response_text = "\n".join(response_parts)
        return {"ok": True, "response": final_response_text}

   
