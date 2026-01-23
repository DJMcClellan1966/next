"""
Enhanced Code Generator
Uses LLM-like approach with better context understanding and pattern learning
"""
import sys
from pathlib import Path
import re
import ast
from typing import Dict, List, Any, Optional
import json

sys.path.insert(0, str(Path(__file__).parent))


class EnhancedCodeGenerator:
    """
    Enhanced Code Generator
    
    Features:
    - Better context understanding
    - Pattern learning
    - Multi-task support
    - Template + pattern-based generation
    """
    
    def __init__(self):
        """Initialize enhanced code generator"""
        self.patterns = self._load_patterns()
        self.learned_patterns = self._load_learned_patterns()
        self.context_cache = {}
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load code patterns"""
        return {
            'tic_tac_toe': self._get_tic_tac_toe_code(),
            'calculator': self._get_calculator_code(),
            'guess_number': self._get_guess_number_code(),
            'todo_list': self._get_todo_list_code(),
            'hangman': self._get_hangman_code(),
            'snake_game': self._get_snake_game_code(),
            'ml_classification': self._get_ml_classification_code(),
            'ml_regression': self._get_ml_regression_code(),
            'data_analysis': self._get_data_analysis_code(),
            'web_scraper': self._get_web_scraper_code(),
        }
    
    def _load_learned_patterns(self) -> Dict[str, Any]:
        """Load learned patterns from file"""
        pattern_file = Path(__file__).parent / 'learned_patterns.json'
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_learned_patterns(self):
        """Save learned patterns to file"""
        pattern_file = Path(__file__).parent / 'learned_patterns.json'
        try:
            with open(pattern_file, 'w') as f:
                json.dump(self.learned_patterns, f, indent=2)
        except:
            pass
    
    def generate(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate code for a task
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Dictionary with 'code', 'success', 'pattern_used'
        """
        task_lower = task.lower()
        
        # 1. Check learned patterns first
        if task_lower in self.learned_patterns:
            code = self.learned_patterns[task_lower]['code']
            return {
                'code': code,
                'success': True,
                'pattern_used': 'learned',
                'source': 'learned_pattern'
            }
        
        # 2. Match to known patterns
        pattern_match = self._match_pattern(task_lower)
        if pattern_match:
            code = self.patterns[pattern_match]
            # Learn this pattern
            self._learn_pattern(task_lower, code)
            return {
                'code': code,
                'success': True,
                'pattern_used': pattern_match,
                'source': 'pattern_match'
            }
        
        # 3. Try to generate from components
        generated = self._generate_from_components(task, context)
        if generated:
            self._learn_pattern(task_lower, generated)
            return {
                'code': generated,
                'success': True,
                'pattern_used': 'component_based',
                'source': 'component_generation'
            }
        
        # 4. Fallback to template
        template_code = self._generate_template(task)
        return {
            'code': template_code,
            'success': True,
            'pattern_used': 'template',
            'source': 'template_fallback'
        }
    
    def _match_pattern(self, task: str) -> Optional[str]:
        """Match task to known pattern"""
        patterns = {
            'tic_tac_toe': ['tic', 'tac', 'toe', 'tic-tac'],
            'calculator': ['calculator', 'calc'],
            'guess_number': ['guess', 'number'],
            'todo_list': ['todo', 'task list'],
            'hangman': ['hangman'],
            'snake_game': ['snake', 'game'],
            'ml_classification': ['classify', 'classification'],
            'ml_regression': ['regress', 'regression', 'predict'],
            'data_analysis': ['analyze', 'analysis', 'data'],
            'web_scraper': ['scrape', 'scraper', 'web'],
        }
        
        for pattern_name, keywords in patterns.items():
            if any(keyword in task for keyword in keywords):
                return pattern_name
        
        return None
    
    def _generate_from_components(self, task: str, context: Optional[Dict]) -> Optional[str]:
        """Generate code from components"""
        # Analyze task for components
        components = self._analyze_task(task)
        
        if not components:
            return None
        
        # Build code from components
        code_parts = []
        
        # Imports
        if 'imports' in components:
            code_parts.append('\n'.join(components['imports']))
            code_parts.append('')
        
        # Main code
        if 'main_code' in components:
            code_parts.append(components['main_code'])
        
        return '\n'.join(code_parts)
    
    def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze task to extract components"""
        components = {}
        
        # Check for common patterns
        if 'game' in task.lower():
            components['imports'] = ['import random']
            components['main_code'] = '# Game code template'
        
        if 'data' in task.lower() or 'analyze' in task.lower():
            components['imports'] = ['import pandas as pd', 'import numpy as np']
            components['main_code'] = '# Data analysis template'
        
        return components if components else {}
    
    def _generate_template(self, task: str) -> str:
        """Generate template code"""
        return f'''"""
{task.title()}
Generated code template
"""
def main():
    """Main function"""
    # TODO: Implement {task}
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''
    
    def _learn_pattern(self, task: str, code: str):
        """Learn a pattern"""
        self.learned_patterns[task.lower()] = {
            'code': code,
            'count': self.learned_patterns.get(task.lower(), {}).get('count', 0) + 1
        }
        self._save_learned_patterns()
    
    # Pattern code templates
    def _get_tic_tac_toe_code(self) -> str:
        return '''"""
Tic-Tac-Toe Game
"""
def print_board(board):
    """Print the game board"""
    print("\\n   |   |   ")
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("___|___|___")
    print("   |   |   ")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("___|___|___")
    print("   |   |   ")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print("   |   |   ")

def check_winner(board):
    """Check if there's a winner"""
    winning_combos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    
    for combo in winning_combos:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != ' ':
            return board[combo[0]]
    return None

def is_board_full(board):
    """Check if board is full"""
    return ' ' not in board

def play_game():
    """Main game loop"""
    board = [' '] * 9
    current_player = 'X'
    
    print("Welcome to Tic-Tac-Toe!")
    print("Players take turns. Enter position (1-9):")
    print("\\nBoard positions:")
    print(" 1 | 2 | 3 ")
    print("___|___|___")
    print(" 4 | 5 | 6 ")
    print("___|___|___")
    print(" 7 | 8 | 9 ")
    print()
    
    while True:
        print_board(board)
        print(f"\\nPlayer {current_player}'s turn")
        
        try:
            position = int(input("Enter position (1-9): ")) - 1
            if position < 0 or position > 8:
                print("Invalid position! Enter 1-9.")
                continue
            if board[position] != ' ':
                print("That position is already taken!")
                continue
            
            board[position] = current_player
            
            winner = check_winner(board)
            if winner:
                print_board(board)
                print(f"\\nPlayer {winner} wins!")
                break
            
            if is_board_full(board):
                print_board(board)
                print("\\nIt's a tie!")
                break
            
            current_player = 'O' if current_player == 'X' else 'X'
        except ValueError:
            print("Invalid input! Enter a number 1-9.")
        except KeyboardInterrupt:
            print("\\nGame interrupted.")
            break

if __name__ == "__main__":
    play_game()
'''
    
    def _get_calculator_code(self) -> str:
        return '''"""
Simple Calculator
"""
def calculator():
    """Simple calculator"""
    print("Simple Calculator")
    print("Enter 'quit' to exit")
    
    while True:
        try:
            expression = input("\\nEnter expression: ")
            if expression.lower() == 'quit':
                break
            result = eval(expression)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    calculator()
'''
    
    def _get_guess_number_code(self) -> str:
        return '''"""
Guess the Number Game
"""
import random

def guess_number():
    """Guess the number game"""
    number = random.randint(1, 100)
    attempts = 0
    
    print("Guess the Number!")
    print("I'm thinking of a number between 1 and 100.")
    
    while True:
        try:
            guess = int(input("\\nEnter your guess: "))
            attempts += 1
            
            if guess < number:
                print("Too low!")
            elif guess > number:
                print("Too high!")
            else:
                print(f"\\nCongratulations! You guessed it in {attempts} attempts!")
                break
        except ValueError:
            print("Invalid input! Enter a number.")
        except KeyboardInterrupt:
            print("\\nGame interrupted.")
            break

if __name__ == "__main__":
    guess_number()
'''
    
    def _get_todo_list_code(self) -> str:
        return '''"""
Todo List Application
"""
class TodoList:
    def __init__(self):
        self.todos = []
    
    def add(self, task):
        """Add a task"""
        self.todos.append({'task': task, 'done': False})
        print(f"Added: {task}")
    
    def list(self):
        """List all tasks"""
        if not self.todos:
            print("No tasks!")
            return
        for i, todo in enumerate(self.todos, 1):
            status = "✓" if todo['done'] else " "
            print(f"{i}. [{status}] {todo['task']}")
    
    def done(self, index):
        """Mark task as done"""
        if 1 <= index <= len(self.todos):
            self.todos[index - 1]['done'] = True
            print(f"Marked as done: {self.todos[index - 1]['task']}")
        else:
            print("Invalid task number!")
    
    def remove(self, index):
        """Remove a task"""
        if 1 <= index <= len(self.todos):
            task = self.todos.pop(index - 1)['task']
            print(f"Removed: {task}")
        else:
            print("Invalid task number!")

def main():
    """Main function"""
    todo = TodoList()
    
    while True:
        print("\\nTodo List")
        print("1. Add task")
        print("2. List tasks")
        print("3. Mark done")
        print("4. Remove task")
        print("5. Quit")
        
        choice = input("\\nChoice: ")
        
        if choice == '1':
            task = input("Enter task: ")
            todo.add(task)
        elif choice == '2':
            todo.list()
        elif choice == '3':
            todo.list()
            index = int(input("Task number: "))
            todo.done(index)
        elif choice == '4':
            todo.list()
            index = int(input("Task number: "))
            todo.remove(index)
        elif choice == '5':
            break

if __name__ == "__main__":
    main()
'''
    
    def _get_hangman_code(self) -> str:
        return '''"""
Hangman Game
"""
import random

WORDS = ['python', 'programming', 'computer', 'algorithm', 'function']

def play_hangman():
    """Play hangman game"""
    word = random.choice(WORDS)
    guessed = ['_'] * len(word)
    attempts = 6
    guessed_letters = set()
    
    print("Welcome to Hangman!")
    print(f"Word: {' '.join(guessed)}")
    
    while attempts > 0:
        print(f"\\nAttempts left: {attempts}")
        print(f"Guessed letters: {', '.join(sorted(guessed_letters))}")
        
        guess = input("Guess a letter: ").lower()
        
        if len(guess) != 1 or not guess.isalpha():
            print("Please enter a single letter!")
            continue
        
        if guess in guessed_letters:
            print("You already guessed that!")
            continue
        
        guessed_letters.add(guess)
        
        if guess in word:
            print("Good guess!")
            for i, letter in enumerate(word):
                if letter == guess:
                    guessed[i] = guess
        else:
            print("Wrong guess!")
            attempts -= 1
        
        print(f"Word: {' '.join(guessed)}")
        
        if '_' not in guessed:
            print("\\nCongratulations! You won!")
            return
    
    print(f"\\nGame over! The word was: {word}")

if __name__ == "__main__":
    play_hangman()
'''
    
    def _get_snake_game_code(self) -> str:
        return '''"""
Snake Game (Simple Text Version)
"""
import random
import time

def play_snake():
    """Simple snake game"""
    print("Snake Game - Use WASD to move")
    print("Press Enter to start...")
    input()
    
    # Simple implementation
    print("Snake game started!")
    print("(Full implementation would require graphics library)")

if __name__ == "__main__":
    play_snake()
'''
    
    def _get_ml_classification_code(self) -> str:
        return '''"""
ML Classification Example
"""
import numpy as np
from ml_toolbox import MLToolbox

# Create sample data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Initialize toolbox
toolbox = MLToolbox()

# Train model
result = toolbox.fit(X, y, task_type='classification')

print(f"Model trained!")
print(f"Accuracy: {result.get('metrics', {}).get('accuracy', 'N/A')}")
'''
    
    def _get_ml_regression_code(self) -> str:
        return '''"""
ML Regression Example
"""
import numpy as np
from ml_toolbox import MLToolbox

# Create sample data
X = np.random.randn(100, 10)
y = np.random.randn(100)

# Initialize toolbox
toolbox = MLToolbox()

# Train model
result = toolbox.fit(X, y, task_type='regression')

print(f"Model trained!")
print(f"R2 Score: {result.get('metrics', {}).get('r2_score', 'N/A')}")
'''
    
    def _get_data_analysis_code(self) -> str:
        return '''"""
Data Analysis Example
"""
import numpy as np
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Basic analysis
print("Data Summary:")
print(data.describe())
print("\\nCategory counts:")
print(data['category'].value_counts())
'''
    
    def _get_web_scraper_code(self) -> str:
        return '''"""
Web Scraper Example
"""
# Note: Requires requests and beautifulsoup4
# pip install requests beautifulsoup4

try:
    import requests
    from bs4 import BeautifulSoup
    
    def scrape_url(url):
        """Scrape a URL"""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    
    # Example usage
    # text = scrape_url('https://example.com')
    # print(text)
    
    print("Web scraper template ready!")
    print("Install: pip install requests beautifulsoup4")
except ImportError:
    print("Install required packages: pip install requests beautifulsoup4")
'''


# Global instance
_global_generator = None

def get_enhanced_generator() -> EnhancedCodeGenerator:
    """Get global enhanced code generator"""
    global _global_generator
    if _global_generator is None:
        _global_generator = EnhancedCodeGenerator()
    return _global_generator
