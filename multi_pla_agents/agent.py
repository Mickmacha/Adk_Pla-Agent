from google.adk.agents import Agent
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio

classification_agent = LlmAgent(
    name="ClassificationAgent",
    model="gemini-1.5-flash", # Or gemini-1.5-pro, depending on your needs
    description="An agent that classifies a students profile into a specific tech specialization ",
    instruction="""You are an expert career counselor and skills analyzer.
    Analyze comprehensive student profiles and classify their primary career focus based on:

    - Technical skills (programming languages, software proficiency, technical skills)
    - Professional experience (job titles, responsibilities, employers)
    - Educational background (degrees, fields of study, institutions)
    - Career goals (short-term, long-term aspirations)
    - Certifications and achievements
    - Learning preferences and skills to acquire

    Consider the student's overall profile holistically. Look for patterns across:
    - Current skill levels vs. desired skills
    - Academic background vs. career aspirations
    - Professional experience vs. future goals

    Return ONLY a specific classification category such as:
    - "AI/Machine Learning Engineer"
    - "Full-Stack Web Developer" 
    - "Data Scientist"
    - "DevOps Engineer"
    - "Mobile App Developer"
    - "Cybersecurity Specialist"
    - "Product Manager"
    - "UI/UX Designer"
    - "Backend Developer"
    - "Frontend Developer"
    - "Cloud Solutions Architect"
    - "Business Analyst"
    - "Digital Marketing Specialist"
    - etc.

    Be specific and precise based on the strongest indicators in their profile.
    """,
    output_key="career_classification"  # This will store the classification in the session state
)

# 2. Define the Recommendation Agent as an LlmAgent
recommendation_agent = LlmAgent(
    name="RecommendationAgent",
    model="gemini-1.5-flash", # Or gemini-1.5-pro
    description="An agents that recommends courses based on the student's career classification",
    instruction="""You are a personalized learning path advisor. Based on a student's
    comprehensive profile and the provided career classification, provide 3-4 specific, actionable learning
    recommendations that will accelerate their career development.

    The student's career classification is: {career_classification}

    Consider their:
    - Current skill gaps vs. desired skills
    - Learning style preferences
    - Career timeline (short-term vs long-term goals)
    - Current experience level
    - Preferred learning methods

    Format your response as a numbered list with concrete, practical suggestions.
    Each recommendation should be:
    1. Specific (mention exact technologies, courses, or skills)
    2. Actionable (something they can start immediately)
    3. Relevant to their career path
    4. Progressive (building on their current level)

    Examples of good recommendations:
    - "Complete AWS Cloud Practitioner certification within 3 months to strengthen cloud fundamentals"
    - "Build 2-3 React projects with TypeScript to demonstrate frontend proficiency"
    - "Practice data structures and algorithms on LeetCode 30 minutes daily for technical interviews"
    """,
    # This agent will implicitly use the 'career_classification' from the session state
    # due to its presence in the instruction.
    # We can also explicitly set an output_key if this agent's output needs to be
    # used by subsequent agents, but for the final output, it's not strictly necessary.
)

personalization_workflow = SequentialAgent(
    name="CareerAdvisorWorkflow",
    sub_agents=[
        classification_agent,
        recommendation_agent
    ],
    description="A workflow that classifies a student's career focus and then provides personalized learning recommendations."
)

root_agent = personalization_workflow