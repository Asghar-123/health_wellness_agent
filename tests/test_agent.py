import pytest
from agent import HealthPlannerAgent
from context import UserSessionContext


@pytest.fixture
def health_planner_agent():
    return HealthPlannerAgent()


@pytest.mark.asyncio
async def test_dynamic_instructions_casual_chat_friendly(health_planner_agent):
    user_input = "Hello, be friendly"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "very friendly and encouraging tone" in response

@pytest.mark.asyncio
async def test_dynamic_instructions_casual_chat_serious(health_planner_agent):
    user_input = "Hi there, be serious"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "very serious and direct tone" in response

@pytest.mark.asyncio
async def test_dynamic_instructions_casual_chat_funny(health_planner_agent):
    user_input = "How are you, be funny"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "humorous" in response

@pytest.mark.asyncio
async def test_dynamic_instructions_casual_chat_default(health_planner_agent):
    user_input = "Thanks for your help"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "specialized AI assistant with expertise in health, biology, and medical queries" in response
    assert "very friendly and encouraging tone" not in response
    assert "very serious and direct tone" not in response
    assert "humorous" not in response

@pytest.mark.asyncio
async def test_dynamic_instructions_fallback_friendly(health_planner_agent):
    user_input = "I want to explore Mars, be friendly"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "very friendly and encouraging tone" in response

@pytest.mark.asyncio
async def test_dynamic_instructions_fallback_serious(health_planner_agent):
    user_input = "Tell me about quantum physics, be serious"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "very serious and direct tone" in response

@pytest.mark.asyncio
async def test_dynamic_instructions_fallback_funny(health_planner_agent):
    user_input = "What's the meaning of life, be funny"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "humorous" in response

@pytest.mark.asyncio
async def test_dynamic_instructions_fallback_default(health_planner_agent):
    user_input = "What is the capital of France?"
    ctx = UserSessionContext()
    response = await health_planner_agent.run(user_input, ctx)
    assert "specialized AI assistant with expertise in health, biology, and medical queries" in response
    assert "very friendly and encouraging tone" not in response
    assert "very serious and direct tone" not in response
    assert "humorous" not in response