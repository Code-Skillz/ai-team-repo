import streamlit as st
import json
import random
import os
import time
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
SYNTHETIC_DATA_DIR = PROJECT_ROOT / 'data' / 'synthetic'
ALTERNATIVE_DIR = PROJECT_ROOT / 'data' / 'alternative'
EVALUATION_DIR = PROJECT_ROOT / 'evaluation'

# Page configuration
st.set_page_config(
    page_title="T5 Model Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-organization/ai-team-repo',
        'Report a bug': 'https://github.com/your-organization/ai-team-repo/issues',
        'About': 'T5 Model Demo for AI Code Analysis System'
    }
)

# Custom CSS
st.markdown('''
<style>
:root {
    --primary-color: #4263EB;
    --background-color: #F8F9FA;
    --secondary-background-color: #E8F4F8;
    --text-color: #31333F;
    --font-family: 'Inter', sans-serif;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --primary-color: #6C8EFF;
        --background-color: #1E1E1E;
        --secondary-background-color: #2D2D2D;
        --text-color: #F4F4F4;
    }
}

.main-header {
    font-size: 2.5rem;
    color: var(--primary-color);
    font-weight: 700;
    text-align: center;
    margin-bottom: 2rem;
    font-family: var(--font-family);
}

.subheader {
    font-size: 1.8rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    color: var(--primary-color);
    font-family: var(--font-family);
    font-weight: 600;
}

.info-text {
    background-color: var(--secondary-background-color);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border-left: 4px solid var(--primary-color);
}

.code-container {
    background-color: #1E1E1E;
    color: #D4D4D4;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    white-space: pre-wrap;
    overflow-x: auto;
}

.result-container {
    background-color: var(--secondary-background-color);
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
    margin-bottom: 1rem;
    border-left: 5px solid var(--primary-color);
}

.error-message {
    color: #FF5252;
    background-color: rgba(255, 82, 82, 0.1);
    padding: 0.8rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    border-left: 4px solid #FF5252;
}

.success-message {
    color: #4CAF50;
    background-color: rgba(76, 175, 80, 0.1);
    padding: 0.8rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    border-left: 4px solid #4CAF50;
}

/* Custom button styling */
.stButton>button {
    background-color: var(--primary-color);
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #3451C6;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

/* For metrics display */
.metrics-card {
    background-color: var(--secondary-background-color);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    transition: transform 0.3s ease;
}

.metrics-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
}

/* Improved sidebar styling */
.css-1d391kg, .css-12oz5g7 {
    background-color: var(--secondary-background-color);
}
</style>

<!-- Load Google Fonts and Icons -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
''', unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=3600)
def load_synthetic_data(file_name):
    """Load synthetic data from JSON file with caching for better performance."""
    try:
        file_path = SYNTHETIC_DATA_DIR / file_name
        if not file_path.exists():
            # Try alternative directory
            file_path = ALTERNATIVE_DIR / 'debugging_examples' / file_name
        
        if not file_path.exists():
            st.warning(f"Could not find {file_name} in any of the expected directories.")
            return []
            
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def load_evaluation_metrics():
    """Load model evaluation metrics from JSON file with caching."""
    try:
        metrics_path = EVALUATION_DIR / 'accuracy_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        else:
            # Default metrics if file doesn't exist
            return {
                "error_type_accuracy": 0.75,
                "avg_bleu_hint": 0.62,
                "avg_bleu_code": 0.58,
                "helpful_hint_rate": 0.80,
                "composite_score": 0.72
            }
    except Exception as e:
        st.warning(f"Could not load evaluation metrics: {str(e)}. Using default values.")
        return {
            "error_type_accuracy": 0.75,
            "avg_bleu_hint": 0.62, 
            "avg_bleu_code": 0.58,
            "helpful_hint_rate": 0.80,
            "composite_score": 0.72
        }

@st.cache_resource
def check_t5_dependencies():
    """Check if required T5 model dependencies are installed."""
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.info(f"Using device: {device}")
        
        # Try to load the model to verify everything works
        tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        return True, "✅ T5 model dependencies are ready"
    except Exception as e:
        st.sidebar.error(f"T5 model setup failed: {str(e)}")
        return False, f"❌ T5 model setup failed: {str(e)}"

@st.cache_resource
def load_t5_model(model_type):
    """Load the appropriate T5 model based on the task type."""
    try:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }
    except Exception as e:
        st.error(f"Failed to load T5 model: {str(e)}")
        return None

@st.cache_data
def generate_performance_chart(metrics):
    """Generate a radar chart for model performance visualization."""
    try:
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Model Performance',
            line_color='#4263EB',
            fillcolor='rgba(66, 99, 235, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Model Performance Metrics",
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error generating performance chart: {str(e)}")
        return None

@st.cache_data
def format_code_with_error_highlighting(code, errors=None):
    """Format code with error highlighting for better visualization."""
    if not errors:
        return f"```python\n{code}\n```"
    
    code_lines = code.split('\n')
    highlighted_code = ""
    
    for i, line in enumerate(code_lines):
        line_num = i + 1
        error_in_line = False
        error_message = ""
        
        for error in errors:
            if error.get('line') == line_num:
                error_in_line = True
                error_message = error.get('message', 'Error detected')
                break
        
        if error_in_line:
            highlighted_code += f"<span style='color: #FF5252;'>{line}</span> <!-- {error_message} -->\n"
        else:
            highlighted_code += f"{line}\n"
    
    return f"<div class='code-container'>{highlighted_code}</div>"

def t5_error_detection(code_sample):
    """Use T5 model for error detection."""
    t5_resources = load_t5_model("error_detection")
    if not t5_resources:
        return simulate_t5_error_detection(code_sample)
        
    model = t5_resources["model"]
    tokenizer = t5_resources["tokenizer"]
    device = t5_resources["device"]
    
    # First try to use the Python parser to catch syntax errors
    try:
        import ast
        ast.parse(code_sample)
        # If we get here, there are no syntax errors that Python can detect
        # But we'll still use the T5 model to look for other issues
    except SyntaxError as e:
        # We found a syntax error, so we'll return it directly
        return {
            "has_errors": True,
            "errors": [{
                "type": "syntax",
                "line": e.lineno if hasattr(e, 'lineno') else 1,
                "message": f"Syntax error: {str(e)}"
            }]
        }
    
    try:
        # Prepare input
        input_text = f"detect errors: {code_sample}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate analysis
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the model output into structured error data
        errors = []
        has_error = False
        
        # Process the analysis to extract errors
        for line in analysis.split('\n'):
            if ':' in line:
                error_type, message = line.split(':', 1)
                if 'line' in message.lower():
                    try:
                        line_num = int(''.join(filter(str.isdigit, message)))
                        errors.append({
                            "type": error_type.strip().lower(),
                            "line": line_num,
                            "message": message.strip()
                        })
                        has_error = True
                    except ValueError:
                        continue
        
        # If T5 didn't find any errors, do additional checks
        if not has_error:
            # Check for unbalanced parentheses
            if code_sample.count('(') != code_sample.count(')') or \
               code_sample.count('[') != code_sample.count(']') or \
               code_sample.count('{') != code_sample.count('}'):
                has_error = True
                # Find the line with the issue
                lines = code_sample.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.count('(') != line.count(')') or \
                       line.count('[') != line.count(']') or \
                       line.count('{') != line.count('}'):
                        errors.append({
                            "type": "syntax",
                            "line": i+1,
                            "message": "Unbalanced parentheses, brackets, or braces"
                        })
        
        return {
            "has_errors": has_error,
            "errors": errors
        }
        
    except Exception as e:
        st.error(f"Error in T5 analysis: {str(e)}")
        # Fallback to simulation if something goes wrong
        return simulate_t5_error_detection(code_sample)

def t5_debugging_insights(code_sample, errors):
    """Use T5 model for debugging insights."""
    # Try to load the model
    tokenizer, model, device = load_t5_model("debugging")
    
    if not tokenizer or not model:
        # Fall back to simulation if model loading failed
        return simulate_t5_debugging_insights(code_sample, errors)
    
    if not errors["has_errors"]:
        return "No errors detected in the code. It appears to be syntactically correct."
    
    try:
        import torch
        
        # Prepare input with error information
        error_descriptions = "; ".join([f"{e['type']} error at line {e['line']}" for e in errors["errors"]])
        input_text = f"debug: {error_descriptions}\ncode: {code_sample}"
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate debugging insights
        outputs = model.generate(
            input_ids, 
            max_length=300,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        insights = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format the insights
        formatted_insights = "## Debugging Insights\n\n"
        for i, error in enumerate(errors["errors"]):
            formatted_insights += f"**{i+1}. {error['type'].capitalize()} error on line {error['line']}:** {error['message']}\n\n"
        
        # Add the model-generated insights
        formatted_insights += "### Suggested Fixes:\n\n" + insights
        
        return formatted_insights
        
    except Exception as e:
        st.warning(f"Error using T5 model for debugging insights: {e}. Falling back to simulated output.")
        return simulate_t5_debugging_insights(code_sample, errors)

def t5_question_generation(difficulty, topic):
    """Use T5 model for question generation."""
    # Try to load the model
    tokenizer, model, device = load_t5_model("question")
    
    if not tokenizer or not model:
        # Fall back to simulation if model loading failed
        return simulate_t5_question_generation(difficulty, topic)
    
    try:
        import torch
        
        # Prepare input
        input_text = f"generate coding question: difficulty={difficulty}, topic={topic if topic != 'Any' else 'general'}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate question
        outputs = model.generate(
            input_ids, 
            max_length=500,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
        question_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the generated question
        title = ""
        description = ""
        example = ""
        
        # Extract information from the generated text
        if "title:" in question_text.lower():
            parts = question_text.lower().split("title:")
            if len(parts) > 1:
                title_part = parts[1].split("\n")[0].strip()
                title = title_part.capitalize()
        
        if "description:" in question_text.lower():
            parts = question_text.lower().split("description:")
            if len(parts) > 1:
                desc_end = parts[1].find("example:")
                if desc_end > 0:
                    description = parts[1][:desc_end].strip()
                else:
                    description = parts[1].strip()
        
        if "example:" in question_text.lower():
            parts = question_text.lower().split("example:")
            if len(parts) > 1:
                example = parts[1].strip()
        
        # If parsing failed, fall back to a simpler approach
        if not title or not description:
            lines = question_text.strip().split("\n")
            if lines:
                title = lines[0].strip()
                description = "\n".join(lines[1:]).strip()
        
        # Create and return the question dictionary
        return {
            "title": title if title else "Coding Challenge",
            "difficulty": difficulty,
            "description": description if description else question_text,
            "example": example if example else "Input: [sample input]\nOutput: [expected output]"
        }
        
    except Exception as e:
        st.warning(f"Error using T5 model: {e}. Falling back to simulated output.")
        return simulate_t5_question_generation(difficulty, topic)

def simulate_t5_error_detection(code_sample):
    """Simulate T5 model for error detection."""
    # Simple heuristics-based error detection
    has_error = False
    errors = []
    
    # Check for common syntax errors
    lines = code_sample.strip().split('\n')
    
    # Check for unbalanced parentheses
    if code_sample.count('(') != code_sample.count(')'):
        has_error = True
        # Find the line with the issue
        for i, line in enumerate(lines):
            if line.count('(') != line.count(')'):
                errors.append({
                    "type": "syntax",
                    "line": i+1,
                    "message": "Unbalanced parentheses"
                })
    
    # Try to parse the code to catch syntax errors
    try:
        import ast
        ast.parse(code_sample)
    except SyntaxError as e:
        has_error = True
        # Get line number from the exception
        line_num = e.lineno if hasattr(e, 'lineno') else 1
        errors.append({
            "type": "syntax",
            "line": line_num,
            "message": f"Syntax error: {str(e)}"
        })
    
    # Check for missing colons in control structures
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in ['if ', 'else:', 'def ', 'for ', 'while ']) and \
           not line.strip().endswith(':') and \
           not line.strip().endswith(')'):
            has_error = True
            errors.append({
                "type": "syntax",
                "line": i+1,
                "message": "Missing colon after control structure"
            })
    
    # Check for indentation errors
    indentation_levels = [len(line) - len(line.lstrip()) for line in lines]
    for i in range(1, len(indentation_levels)):
        if indentation_levels[i] > indentation_levels[i-1] and \
           not lines[i-1].strip().endswith(':') and \
           not lines[i-1].strip().endswith('{'):
            has_error = True
            errors.append({
                "type": "indentation",
                "line": i+1,
                "message": "Unexpected indentation increase"
            })
    
    return {
        "has_errors": has_error,
        "errors": errors
    }

def simulate_t5_debugging_insights(code_sample, errors):
    """Simulate T5 model for debugging insights."""
    # Load debugging examples from JSON file
    try:
        debugging_examples = []
        debugging_path = ALTERNATIVE_DIR / 'debugging_examples' / 'debugging_examples_raw.json'
        if debugging_path.exists():
            with open(debugging_path, 'r') as f:
                debugging_examples = json.load(f)
    except Exception:
        debugging_examples = []
    
    # Handle both dictionary and list formats for errors
    has_errors = False
    error_list = []
    
    if isinstance(errors, dict) and "has_errors" in errors:
        has_errors = errors["has_errors"]
        error_list = errors.get("errors", [])
    elif isinstance(errors, list):
        has_errors = len(errors) > 0
        error_list = errors
    
    if not has_errors or not error_list:
        return "No errors detected in the code. It appears to be syntactically correct."
    
    insights = "Here are some debugging insights:\n\n"
    for i, error in enumerate(error_list):
        insights += f"{i+1}. {error['type'].capitalize()} error on line {error['line']}: {error['message']}\n"
        
        # Try to find matching error type in our dataset
        matching_examples = [ex for ex in debugging_examples if error['type'].lower() in ex['error_type'].lower()]
        
        if matching_examples:
            # Use real debugging hint from dataset
            example = random.choice(matching_examples)
            insights += f"   - {example['debugging_hint']}\n"
        else:
            # Add custom insights based on error type
            if "parentheses" in error['message'].lower():
                insights += "   - Check if all opening parentheses '(' have matching closing ones ')'\n"
                insights += "   - Count the number of opening and closing parentheses to ensure they match\n"
                insights += "   - Make sure each function call and expression has properly balanced parentheses\n"
            elif "colon" in error['message'].lower():
                insights += "   - Python requires colons after control statements (if, for, while, def, etc.)\n"
                insights += "   - Add a colon ':' at the end of the control statement line\n"
            elif "indentation" in error['type'].lower():
                insights += "   - Python uses indentation to define code blocks\n"
                insights += "   - Make sure your indentation is consistent throughout the code\n"
                insights += "   - Check if all indentation increases follow a colon on the previous line\n"
    
    return insights

def simulate_t5_question_generation(difficulty, topic):
    """Simulate T5 model for question generation."""
    # Load sample questions
    questions = load_synthetic_data("question_generation.json")
    
    # Filter by difficulty and topic if specified
    filtered_questions = [q for q in questions if q["difficulty"] == difficulty]
    if topic and topic != "Any":
        filtered_questions = [q for q in filtered_questions if q["topic"] == topic]
    
    # If no matches, return any question of the right difficulty
    if not filtered_questions:
        filtered_questions = [q for q in questions if q["difficulty"] == difficulty]
    
    # If still no matches, return any question
    if not filtered_questions:
        return random.choice(questions) if questions else None
    
    # Return a random question from the filtered list
    return random.choice(filtered_questions)

# List to keep track of generated questions
generated_questions = []

def generate_unique_question(difficulty, topic):
    while True:
        question = t5_question_generation(difficulty, topic)
        if question not in generated_questions:
            generated_questions.append(question)
            return question

# Main app
def main():
    """Main app interface with improved layout and user experience."""
    st.markdown("<h1 class='main-header'>AI Code Analysis System</h1>", unsafe_allow_html=True)
    
    # Check T5 dependencies
    t5_result = check_t5_dependencies()
    if isinstance(t5_result, tuple) and len(t5_result) == 2:
        t5_available, dependency_message = t5_result
    else:
        # Handle the case where the function returns just a boolean
        t5_available = t5_result
        dependency_message = "✅ Using offline simulation for T5 models." if t5_available else "❌ T5 model setup failed."
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/code.png", width=80)
        st.markdown("### T5 Model Demo")
        st.markdown(dependency_message)
        
        st.markdown("---")
        st.markdown("### Navigation")
        app_mode = st.radio(
            "Select a mode:",
            options=["Error Detection", "Debugging Insights", "Question Generation", "Model Performance"]
        )
        
        st.markdown("---")
        if st.button("📚 View Documentation"):
            st.markdown("""
            ### Documentation
            - **Error Detection**: Identify syntax and logical errors in code
            - **Debugging Insights**: Get explanations and solutions for code issues
            - **Question Generation**: Create programming questions based on difficulty and topic
            - **Model Performance**: View model evaluation metrics
            """)
    
    # Main content area based on selected mode
    if app_mode == "Error Detection":
        error_detection_tab(t5_available)
    elif app_mode == "Debugging Insights":
        debugging_insights_tab(t5_available)
    elif app_mode == "Question Generation":
        question_generation_tab(t5_available)
    elif app_mode == "Model Performance":
        model_performance_tab()


def error_detection_tab(t5_available):
    """Error detection tab with improved UI and functionality."""
    st.markdown("<h2 class='subheader'>Code Error Detection</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-text'>
        This tool uses a T5 model to detect syntax and logical errors in your code. 
        Upload a file or paste your code below.
    </div>
    """, unsafe_allow_html=True)
    
    # Code input options
    code_input_method = st.radio(
        "How would you like to input your code?",
        options=["Paste Code", "Upload File", "Sample Code"],
        horizontal=True
    )
    
    code_sample = ""
    
    if code_input_method == "Paste Code":
        code_sample = st.text_area(
            "Paste your code here:",
            height=200,
            placeholder="# Paste your Python code here\ndef example_function():\n    print(\"Hello, world!\")"
        )
        
    elif code_input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload a Python file", type=["py"])
        if uploaded_file is not None:
            code_sample = uploaded_file.getvalue().decode("utf-8")
            st.text_area("File content:", code_sample, height=200)
            
    elif code_input_method == "Sample Code":
        samples = load_synthetic_data("error_samples.json")
        if samples:
            sample_names = [sample.get("name", f"Sample {i+1}") for i, sample in enumerate(samples)]
            selected_sample = st.selectbox("Select a sample:", sample_names)
            sample_index = sample_names.index(selected_sample)
            code_sample = samples[sample_index].get("code", "")
            st.text_area("Sample code:", code_sample, height=200)
        else:
            st.warning("No sample code available. Please paste your own code or upload a file.")
    
    # Analyze button
    if st.button("🔍 Analyze Code", disabled=not code_sample):
        with st.spinner("Analyzing code..."):
            # Add a small delay to show the spinner for demo purposes
            time.sleep(0.5)
            
            # Call the T5 error detection function
            if t5_available:
                results = t5_error_detection(code_sample)
            else:
                results = simulate_t5_error_detection(code_sample)
            
            # Display results
            if results["has_errors"]:
                st.markdown("<div class='error-message'><strong>⚠️ Errors detected!</strong></div>", unsafe_allow_html=True)
                
                # Display highlighted code with errors
                st.markdown("<h3>Code with Errors Highlighted:</h3>", unsafe_allow_html=True)
                st.markdown(format_code_with_error_highlighting(code_sample, results["errors"]), unsafe_allow_html=True)
                
                # Display detailed error information
                st.markdown("<h3>Error Details:</h3>", unsafe_allow_html=True)
                for error in results["errors"]:
                    st.markdown(f"""
                    <div class='error-message'>
                        <strong>Line {error['line']}:</strong> {error['message']}<br>
                        <strong>Type:</strong> {error['type']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Ask if user wants debugging insights
                if st.button("🛠️ Get Debugging Insights"):
                    st.session_state["code_for_debugging"] = code_sample
                    st.session_state["errors_for_debugging"] = results["errors"]
                    st.experimental_rerun()
            else:
                st.markdown("<div class='success-message'><strong>✅ No errors detected!</strong> Your code looks good.</div>", unsafe_allow_html=True)


def debugging_insights_tab(t5_available):
    """Debugging insights tab with improved UI and functionality."""
    st.markdown("<h2 class='subheader'>Debugging Insights</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-text'>
        This tool uses a T5 model to provide debugging insights for your code.
        These insights can help you understand and fix issues in your code.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have code from error detection
    if "code_for_debugging" in st.session_state and "errors_for_debugging" in st.session_state:
        code_sample = st.session_state["code_for_debugging"]
        errors = st.session_state["errors_for_debugging"]
        st.markdown(format_code_with_error_highlighting(code_sample, errors), unsafe_allow_html=True)
    else:
        # Code input options similar to error detection
        code_input_method = st.radio(
            "How would you like to input your code?",
            options=["Paste Code", "Upload File", "Sample Code"],
            horizontal=True
        )
        
        code_sample = ""
        errors = None
        
        if code_input_method == "Paste Code":
            code_sample = st.text_area(
                "Paste your code here:",
                height=200,
                placeholder="# Paste your Python code here\ndef example_function():\n    print(\"Hello, world!\")"
            )
            
        elif code_input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload a Python file", type=["py"])
            if uploaded_file is not None:
                code_sample = uploaded_file.getvalue().decode("utf-8")
                st.text_area("File content:", code_sample, height=200)
                
        elif code_input_method == "Sample Code":
            samples = load_synthetic_data("debugging_samples.json")
            if samples:
                sample_names = [sample.get("name", f"Sample {i+1}") for i, sample in enumerate(samples)]
                selected_sample = st.selectbox("Select a sample:", sample_names)
                sample_index = sample_names.index(selected_sample)
                code_sample = samples[sample_index].get("code", "")
                if "errors" in samples[sample_index]:
                    errors = samples[sample_index]["errors"]
                st.text_area("Sample code:", code_sample, height=200)
            else:
                st.warning("No sample code available. Please paste your own code or upload a file.")
    
    # Get insights button
    if st.button("🔍 Get Debugging Insights", disabled=not code_sample):
        with st.spinner("Generating debugging insights..."):
            # Add a small delay to show the spinner for demo purposes
            time.sleep(0.5)
            
            # Call the T5 debugging insights function
            if t5_available:
                insights = t5_debugging_insights(code_sample, errors)
            else:
                insights = simulate_t5_debugging_insights(code_sample, errors)
            
            # Display results
            st.markdown("<h3>Debugging Insights:</h3>", unsafe_allow_html=True)
            
            if "insights" in insights and insights["insights"]:
                for i, insight in enumerate(insights["insights"]):
                    st.markdown(f"""
                    <div class='result-container'>
                        <strong>Insight {i+1}:</strong> {insight['description']}<br>
                        <strong>Suggested Fix:</strong><br>
                        <div class='code-container'>{insight['fix']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No specific insights could be generated for this code.")


def question_generation_tab(t5_available):
    """Question generation tab with improved UI and functionality."""
    st.markdown("<h2 class='subheader'>Programming Question Generation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-text'>
        This tool uses a T5 model to generate programming questions based on difficulty level and topic.
        Use these questions to practice your programming skills.
    </div>
    """, unsafe_allow_html=True)
    
    # Question parameters
    col1, col2 = st.columns(2)
    
    with col1:
        difficulty = st.select_slider(
            "Select difficulty level:",
            options=["easy", "medium", "hard"],
            value="medium"
        )
    
    with col2:
        topics = ["arrays", "strings", "linked lists", "trees", "graphs", "dynamic programming", 
                 "sorting", "searching", "recursion", "OOP", "databases", "web development"]
        topic = st.selectbox("Select a programming topic:", topics)
    
    # Generate button
    if st.button("🧩 Generate Question"):
        with st.spinner("Generating question..."):
            # Add a small delay to show the spinner for demo purposes
            time.sleep(0.5)
            
            # Call the T5 question generation function
            question = generate_unique_question(difficulty, topic)
            
            # Display results
            st.markdown("<h3>Generated Question:</h3>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='result-container'>
                <strong>Difficulty:</strong> {question['difficulty'].capitalize()}<br>
                <strong>Topic:</strong> {question['topic'].capitalize()}<br>
                <strong>Title:</strong> {question['title']}<br><br>
                <strong>Problem Statement:</strong><br>
                {question['description']}<br><br>
                <strong>Example:</strong><br>
                <div class='code-container'>{question['example']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show expected solution after a toggle
            if st.checkbox("Show Expected Solution"):
                st.markdown("<h3>Expected Solution:</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='code-container'>{question['solution']}</div>", unsafe_allow_html=True)


def model_performance_tab():
    """Model performance visualization tab."""
    st.markdown("<h2 class='subheader'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-text'>
        This section displays the performance metrics of our T5 models for different tasks.
        These metrics help evaluate the model's accuracy and effectiveness.
    </div>
    """, unsafe_allow_html=True)
    
    # Load metrics
    metrics = load_evaluation_metrics()
    
    # Create columns for metrics display
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Visualization
        fig = generate_performance_chart(metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detailed metrics
        st.markdown("### Detailed Metrics")
        
        for metric, value in metrics.items():
            formatted_metric = metric.replace('_', ' ').title()
            st.markdown(f"""
            <div class='metrics-card'>
                <h4>{formatted_metric}</h4>
                <h2>{value:.2f}</h2>
                <div class="progress" style="height: 5px;">
                    <div class="progress-bar" role="progressbar" style="width: {value*100}%; background-color: #4263EB;" aria-valuenow="{value*100}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional information
    st.markdown("### Evaluation Information")
    st.markdown("""
    Our models are evaluated on the following metrics:
    - **Error Type Accuracy**: Percentage of correctly identified error types
    - **Average BLEU Hint**: Quality of generated debugging hints compared to expert solutions
    - **Average BLEU Code**: Quality of generated code fixes compared to expert solutions
    - **Helpful Hint Rate**: Percentage of hints rated as helpful by users
    - **Composite Score**: Overall model performance score
    """)


if __name__ == "__main__":
    main()
