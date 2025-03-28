# AI Team - GitHub Organization

## 📌 Objective
The AI Team is dedicated to building an intelligent system that enhances the coding experience by analyzing submitted code, detecting errors, and providing logical insights rather than direct answers. Our model should:
- Identify syntax and logical errors in code.
- Provide meaningful debugging insights to help users understand their mistakes.
- Generate coding questions based on the user's skill level to encourage learning.
- Seamlessly integrate with the website for real-time interaction.

## 🚀 Project Overview
We are developing an AI-powered system that:
- Identifies errors in code submissions.
- Provides logical reasoning to help users debug their code.
- Generates new coding challenges tailored to the user’s proficiency level.
- Integrates smoothly with the main platform (similar to LeetCode or HackerRank).

## 🏗️ Current Scope
- **Model Selection & Training:** Developing and fine-tuning an T5 model for code analysis.
- **Dataset Collection:** Gathering a comprehensive dataset for training the model.
- **Error Detection Mechanism:** Implementing techniques to identify syntax and logical errors.
- **Insightful Feedback System:** Designing a system that provides meaningful debugging insights.
- **Adaptive Question Generation:** Implementing a model that generates coding challenges suited to the user's skill level.
- **Integration with Web Platform:** Ensuring smooth API communication with the main website.
- **T5 Model Integration:** Integrating the T5 model for unified text-to-text tasks, including error detection, debugging insights, and question generation.

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Machine Learning Frameworks:** PyTorch / TensorFlow
- **Model:** Custom-trained AI model (Exploring options like Mistral AI, CodeBERT, or other LLMs), T5 model for text-to-text tasks
- **Dataset:** Open-source coding datasets + custom-labeled data
- **Deployment:** Docker, FastAPI (for API), AWS/GCP for hosting

## 📂 Repository Structure
```
📁 ai-team-repo/
 ├── 📁 data/        # Training datasets
 ├── 📁 models/      # Trained models & configurations
 ├── 📁 src/         # Model training & inference scripts
 ├── 📁 api/         # API implementation for integration
 ├── 📄 README.md    # Project documentation
 ├── 📄 requirements.txt  # Dependencies
 ├── 📄 .gitignore   # Ignored files
 ├── 📁 notebooks/   # Jupyter notebooks for exploration
 ├── 📁 t5_models/   # T5 model implementations
 ├── 📁 t5_training/ # T5 model training utilities
```

## 🤝 Collaboration & Workflow
1. **Cloning the Repository:**  
   ```bash
   git clone https://github.com/your-organization/ai-team-repo.git
   ```
2. **Creating a Branch:**  
   ```bash
   git checkout -b feature-branch-name
   ```
3. **Committing Changes:**  
   ```bash
   git add .
   git commit -m "Your commit message"
   ```
4. **Pushing Changes:**  
   ```bash
   git push origin feature-branch-name
   ```
5. **Creating a Pull Request:**  
   - Navigate to the repository on GitHub.
   - Click on "Pull Requests" → "New Pull Request".
   - Select the branch and submit for review.

## 📅 Weekly Goals
We follow an iterative approach with a weekly roadmap to track our progress. Updates will be shared regularly in the GitHub issues section.

## 📢 Communication
- **Primary Discussion:** GitHub Issues & Pull Requests
- **Team Updates & Announcements:** Group chat / Slack / Discord
- **Meetings & Sync Calls:** Weekly check-ins

## 🙌 Contributing
All contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide before submitting any code.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## T5 Model Details
The T5 models are implemented in `t5_models.py` and provide the following functionality:
- **Error Detection**: Identifies syntax and logical errors in code
- **Debugging Insights**: Provides explanations and solutions for code issues
- **Question Generation**: Creates programming questions based on difficulty and topic

T5 models are trained using the `train_t5_model` function in `t5_training.py`. The training process converts each task into a text-to-text format:
- **Error Detection**: Input: "detect errors: {code}", Output: "has_errors: yes/no"
- **Debugging Insights**: Input: "explain code: {code}", Output: Explanation text
- **Question Generation**: Input: "generate question: difficulty={difficulty} topic={topic}", Output: Question text

## AI Code Assistant with T5 Models

An interactive application that leverages T5 language models for code analysis, debugging, and question generation.

## Features

- **Error Detection**: Identifies syntax and logic errors in code
- **Debugging Insights**: Provides intelligent debugging suggestions 
- **Question Generation**: Creates coding challenges based on topic and difficulty
- **Code Highlighting**: Displays code with error highlighting
- **Performance Visualization**: Shows model performance metrics

## Setup Options

This application is designed to work in two modes:

### Option 1: Offline Simulation (No Dependencies Required)

The application works out-of-the-box with simulated T5 model behavior. This mode doesn't require any additional dependencies beyond the basic requirements and is perfect for demonstration purposes.

### Option 2: Actual T5 Models (Advanced)

For the best experience, you can use actual pre-trained T5 models. This requires additional setup:

1. **Create a Python virtual environment**:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install core requirements**:
   ```
   pip install streamlit torch transformers
   ```

3. **Install SentencePiece**:
   - SentencePiece is required for T5 tokenization but can be challenging to install on Windows
   - **Option A**: Try installing with pip:
     ```
     pip install sentencepiece
     ```
   - **Option B**: Use conda (recommended for Windows):
     ```
     conda install -c conda-forge sentencepiece
     ```

## Running the Application

```
streamlit run app.py
```

The application will automatically:
1. Try to use actual T5 models if all dependencies are available
2. Fall back to simulation mode if any dependencies are missing

## Troubleshooting

- **SentencePiece installation issues**: This is a common problem on Windows. If you can't install it through pip or conda, the app will still work with simulated models.
- **"CUDA out of memory"**: If you encounter GPU memory issues, try running on CPU by setting the `CUDA_VISIBLE_DEVICES` environment variable to an empty string.

## Future Improvements

- Fine-tuning T5 models specifically for code tasks
- Adding more code languages beyond Python
- Supporting larger context windows for analyzing complete files

## Integration with Django
The AI Code Analysis System can be integrated with Django applications as follows:
1. **Install the package**:
   ```bash
   pip install -e /path/to/ai-code-analysis
   ```
2. **Configure Django settings**:
   ```python
   # settings.py
   AI_CODE_ANALYSIS = {
       'USE_T5_MODELS': True,
       'MODEL_DIR': '/path/to/models',
   }
   ```
3. **Use in views**:
   ```python
   # views.py
   from ai_code_analysis.t5_models import load_t5_model
   
   def analyze_code_view(request):
       code = request.POST.get('code')
       model = load_t5_model('error_detection')
       result = model.detect_errors(code)
       return JsonResponse(result)
   ```

---

Let's build something impactful together! 🚀
