from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key exists
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Use stable version
    temperature=0.1,
    google_api_key=api_key
)

# Updated prompt templates without target column
NULL_STRATEGY_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful data analyst. Given a user's context and null value analysis, 
suggest appropriate strategies for handling missing values.

User Goal: {goal}
Null Analysis: {null_analysis}

Provide specific, actionable recommendations for handling these null values.
Focus on data quality improvement and preprocessing best practices.
""")

DUPLICATE_STRATEGY_PROMPT = ChatPromptTemplate.from_template("""
You are a data cleaning expert. Analyze the duplicate values and provide strategy.

User Goal: {goal}
Duplicate Analysis: {duplicate_analysis}

Suggest appropriate strategies for handling duplicate rows.
Consider the impact on data quality and the user's preprocessing goals.
""")

TYPE_STRATEGY_PROMPT = ChatPromptTemplate.from_template("""
You are a data type optimization expert. Analyze data types and suggest improvements.

User Goal: {goal}
Data Types Analysis: {type_analysis}

Provide recommendations for optimizing data types for better performance and accuracy.
Focus on memory efficiency and proper data representation.
""")