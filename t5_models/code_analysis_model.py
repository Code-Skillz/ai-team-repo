from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging
from pathlib import Path
import ast
import re
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAnalysisT5:
    def __init__(self, model_name='t5-small'):
        """Initialize the T5 model for code analysis tasks."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.skill_levels = ['beginner', 'intermediate', 'advanced']
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> bool:
        """Load the T5 model and tokenizer."""
        try:
            logger.info(f"Loading {self.model_name} model and tokenizer...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def analyze_syntax(self, code: str) -> Dict[str, Union[str, List[str]]]:
        """Analyze code for syntax errors using Python's AST."""
        try:
            ast.parse(code)
            return {"status": "success", "errors": []}
        except SyntaxError as e:
            return {
                "status": "error",
                "errors": [f"Line {e.lineno}: {e.msg} at position {e.offset}"]
            }
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def detect_logical_patterns(self, code: str) -> List[str]:
        """Detect potential logical issues in code."""
        patterns = []
        
        # Check for common logical issues
        if "=" in code and "==" not in code:
            patterns.append("Possible assignment instead of comparison")
        
        if "while True" in code and "break" not in code:
            patterns.append("Infinite loop detected without break condition")
        
        if "except:" in code and "Exception" not in code:
            patterns.append("Bare except clause detected - consider catching specific exceptions")
        
        if "return" in code and "else:" not in code and "if" in code:
            patterns.append("Missing else clause in conditional return statement")
        
        return patterns

    def detect_errors(self, code: str) -> Dict[str, Union[str, List[str]]]:
        """Comprehensive error detection combining syntax and logical analysis."""
        # First check syntax
        syntax_result = self.analyze_syntax(code)
        if syntax_result["status"] == "error":
            return syntax_result

        # Then check logical patterns
        logical_issues = self.detect_logical_patterns(code)
        
        # Use T5 for additional error detection
        prompt = f"detect errors in code: {code}"
        t5_analysis = self._generate_response(prompt, max_length=150)
        
        return {
            "status": "success",
            "syntax_check": "passed",
            "logical_issues": logical_issues,
            "t5_analysis": t5_analysis
        }

    def provide_debugging_insights(self, code: str, errors: Dict[str, Union[str, List[str]]]) -> str:
        """Provide detailed debugging insights based on detected errors."""
        context = f"code: {code}\nerrors: {str(errors)}"
        prompt = f"provide debugging insights and solutions for: {context}"
        return self._generate_response(prompt, max_length=200)

    def generate_practice_question(self, skill_level: str, topic: str) -> Dict[str, str]:
        """Generate a coding question based on skill level and topic."""
        if skill_level.lower() not in self.skill_levels:
            skill_level = "intermediate"
        
        prompt = f"generate {skill_level} level coding question about {topic} with example solution"
        response = self._generate_response(prompt, max_length=300)
        
        # Split response into question and solution
        parts = response.split("Example solution:", 1)
        question = parts[0].strip()
        solution = parts[1].strip() if len(parts) > 1 else ""
        
        return {
            "question": question,
            "solution": solution,
            "difficulty": skill_level,
            "topic": topic
        }

    def _generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate response using the T5 model with improved parameters."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                early_stopping=True,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    def analyze_code_quality(self, code: str) -> Dict[str, Union[str, List[str], Dict]]:
        """Comprehensive code quality analysis."""
        try:
            # Detect errors
            error_analysis = self.detect_errors(code)
            
            # Get debugging insights if errors found
            debugging_insights = self.provide_debugging_insights(
                code, error_analysis
            ) if error_analysis["status"] == "error" or error_analysis.get("logical_issues") else ""
            
            # Code complexity analysis
            complexity_analysis = self._analyze_complexity(code)
            
            return {
                "status": "success",
                "error_analysis": error_analysis,
                "debugging_insights": debugging_insights,
                "complexity_analysis": complexity_analysis
            }
        except Exception as e:
            logger.error(f"Error in code quality analysis: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _analyze_complexity(self, code: str) -> Dict[str, Union[int, str]]:
        """Analyze code complexity."""
        try:
            tree = ast.parse(code)
            
            # Count loops and conditionals
            loops = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
            conditionals = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            
            # Estimate complexity
            complexity = "Low"
            if loops + conditionals > 5:
                complexity = "Medium"
            if loops + conditionals > 10:
                complexity = "High"
            
            return {
                "loops": loops,
                "conditionals": conditionals,
                "functions": functions,
                "estimated_complexity": complexity
            }
        except Exception as e:
            logger.error(f"Error analyzing complexity: {str(e)}")
            return {
                "error": str(e)
            }
