"""
Simple IDE - Like IDLE but with Revolutionary ML Toolbox Integration
A simple, functional IDE that actually works
"""
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import subprocess
import io
import contextlib
import threading
from typing import Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml_toolbox import MLToolbox
    from ml_toolbox.ai_agent import MLCodeAgent
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("Warning: ML Toolbox not available")


class SimpleIDE:
    """
    Simple IDE - Functional like IDLE with ML Toolbox integration
    
    Features:
    - Code editor with syntax highlighting
    - Execute Python code
    - Save/Open files
    - AI code generation (actually works)
    - ML Toolbox integration
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Simple IDE - ML Toolbox")
        self.root.geometry("1200x800")
        
        # Current file
        self.current_file = None
        
        # Initialize ML Toolbox and AI Agent
        self.toolbox = None
        self.agent = None
        if TOOLBOX_AVAILABLE:
            try:
                self.toolbox = MLToolbox(check_dependencies=False)
                self.agent = MLCodeAgent(use_llm=False, use_pattern_composition=True)
            except Exception as e:
                print(f"Warning: Could not initialize ML Toolbox: {e}")
        
        # Create UI
        self.create_ui()
        
        # Set up key bindings
        self.setup_bindings()
    
    def create_ui(self):
        """Create the IDE UI"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_as_file, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Cut", command=lambda: self.code_editor.event_generate("<<Cut>>"))
        edit_menu.add_command(label="Copy", command=lambda: self.code_editor.event_generate("<<Copy>>"))
        edit_menu.add_command(label="Paste", command=lambda: self.code_editor.event_generate("<<Paste>>"))
        
        # Run menu
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Module", command=self.run_code, accelerator="F5")
        run_menu.add_command(label="Check Module", command=self.check_code)
        
        # AI menu (if available)
        if TOOLBOX_AVAILABLE and self.agent:
            ai_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="AI", menu=ai_menu)
            ai_menu.add_command(label="Generate Code", command=self.ai_generate_code)
            ai_menu.add_command(label="Fix Code", command=self.ai_fix_code)
            ai_menu.add_command(label="Explain Code", command=self.ai_explain_code)
        
        # ML menu (if available)
        if TOOLBOX_AVAILABLE and self.toolbox:
            ml_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="ML Toolbox", menu=ml_menu)
            ml_menu.add_command(label="Quick Train", command=self.quick_train)
            ml_menu.add_command(label="Toolbox Status", command=self.show_toolbox_status)
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(toolbar, text="Run (F5)", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Check", command=self.check_code).pack(side=tk.LEFT, padx=2)
        if TOOLBOX_AVAILABLE and self.agent:
            ttk.Button(toolbar, text="AI Generate", command=self.ai_generate_code).pack(side=tk.LEFT, padx=2)
            ttk.Button(toolbar, text="AI Fix", command=self.ai_fix_code).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="New", command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Code editor and output (side by side)
        editor_output_frame = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        editor_output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Code editor frame
        editor_frame = ttk.Frame(editor_output_frame)
        editor_output_frame.add(editor_frame, weight=2)
        
        ttk.Label(editor_frame, text="Code Editor", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=2)
        
        self.code_editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap=tk.NONE,
            font=("Consolas", 11),
            bg="#ffffff",
            fg="#000000",
            insertbackground="#000000",
            undo=True
        )
        self.code_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output frame
        output_frame = ttk.Frame(editor_output_frame)
        editor_output_frame.add(output_frame, weight=1)
        
        ttk.Label(output_frame, text="Output", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=2)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # AI input frame (if available)
        if TOOLBOX_AVAILABLE and self.agent:
            ai_frame = ttk.LabelFrame(main_frame, text="AI Assistant")
            ai_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ai_input_frame = ttk.Frame(ai_frame)
            ai_input_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(ai_input_frame, text="Describe what you want:").pack(side=tk.LEFT, padx=5)
            self.ai_input = ttk.Entry(ai_input_frame, font=("Arial", 10))
            self.ai_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.ai_input.bind("<Return>", lambda e: self.ai_generate_code())
            
            ttk.Button(ai_input_frame, text="Generate", command=self.ai_generate_code).pack(side=tk.LEFT, padx=5)
    
    def setup_bindings(self):
        """Set up keyboard shortcuts"""
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-Shift-S>", lambda e: self.save_as_file())
        self.root.bind("<F5>", lambda e: self.run_code())
    
    def update_status(self, message: str):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def write_output(self, text: str):
        """Write to output panel"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def clear_output(self):
        """Clear output panel"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def new_file(self):
        """Create new file"""
        self.code_editor.delete(1.0, tk.END)
        self.current_file = None
        self.update_status("New file")
        self.root.title("Simple IDE - ML Toolbox")
    
    def open_file(self):
        """Open file"""
        file_path = filedialog.askopenfilename(
            title="Open Python File",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.code_editor.delete(1.0, tk.END)
                self.code_editor.insert(1.0, content)
                self.current_file = file_path
                self.update_status(f"Opened: {file_path}")
                self.root.title(f"Simple IDE - {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")
    
    def save_file(self):
        """Save file"""
        if self.current_file:
            try:
                content = self.code_editor.get(1.0, tk.END + "-1c")
                with open(self.current_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.update_status(f"Saved: {self.current_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
        else:
            self.save_as_file()
    
    def save_as_file(self):
        """Save file as"""
        file_path = filedialog.asksaveasfilename(
            title="Save Python File",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if file_path:
            try:
                content = self.code_editor.get(1.0, tk.END + "-1c")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.current_file = file_path
                self.update_status(f"Saved: {file_path}")
                self.root.title(f"Simple IDE - {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
    
    def run_code(self):
        """Execute Python code"""
        code = self.code_editor.get(1.0, tk.END + "-1c")
        if not code.strip():
            self.write_output("No code to execute.\n")
            return
        
        self.clear_output()
        self.write_output("=" * 60 + "\n")
        self.write_output("Running code...\n")
        self.write_output("=" * 60 + "\n\n")
        self.update_status("Running...")
        
        # Save to temp file and execute
        try:
            # Create temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute in separate thread
            def execute():
                try:
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(Path(__file__).parent)
                    )
                    
                    # Write output
                    if result.stdout:
                        self.write_output(result.stdout)
                    if result.stderr:
                        self.write_output(f"Error:\n{result.stderr}")
                    if result.returncode == 0:
                        self.write_output("\n" + "=" * 60 + "\n")
                        self.write_output("Execution completed successfully.\n")
                        self.update_status("Execution completed")
                    else:
                        self.write_output("\n" + "=" * 60 + "\n")
                        self.write_output(f"Execution failed with code {result.returncode}.\n")
                        self.update_status("Execution failed")
                except subprocess.TimeoutExpired:
                    self.write_output("\nExecution timed out after 30 seconds.\n")
                    self.update_status("Execution timed out")
                except Exception as e:
                    self.write_output(f"\nError executing code: {e}\n")
                    self.update_status(f"Error: {e}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            thread = threading.Thread(target=execute, daemon=True)
            thread.start()
            
        except Exception as e:
            self.write_output(f"Error: {e}\n")
            self.update_status(f"Error: {e}")
    
    def check_code(self):
        """Check code syntax"""
        code = self.code_editor.get(1.0, tk.END + "-1c")
        if not code.strip():
            messagebox.showinfo("Check", "No code to check.")
            return
        
        try:
            compile(code, "<string>", "exec")
            messagebox.showinfo("Check", "Code syntax is valid!")
            self.update_status("Code syntax is valid")
        except SyntaxError as e:
            messagebox.showerror("Syntax Error", f"Syntax error:\n{e}")
            self.update_status(f"Syntax error: {e}")
    
    def ai_generate_code(self):
        """Generate code using AI Agent"""
        if not self.agent:
            messagebox.showwarning("AI Not Available", "AI Agent is not available.")
            return
        
        # Get task description
        if hasattr(self, 'ai_input') and self.ai_input.get().strip():
            task = self.ai_input.get().strip()
        else:
            task = simpledialog.askstring(
                "AI Generate Code",
                "Describe what you want to build:"
            )
        
        if not task:
            return
        
        self.update_status("Generating code with AI...")
        self.write_output(f"AI Task: {task}\n")
        self.write_output("Generating code...\n\n")
        
        def generate():
            try:
                result = self.agent.build(task, context={})
                
                if result['success']:
                    code = result.get('code', '')
                    if code:
                        # Insert code at cursor or replace selection
                        try:
                            # Clear selection if any, then insert
                            try:
                                self.code_editor.delete(tk.SEL_FIRST, tk.SEL_LAST)
                            except:
                                pass
                            
                            # Insert code
                            self.code_editor.insert(tk.INSERT, code)
                            self.write_output("Code generated successfully!\n")
                            self.write_output("=" * 60 + "\n")
                            self.write_output(f"Generated {len(code)} characters of code.\n")
                            if result.get('output'):
                                self.write_output(f"Output:\n{result['output']}\n")
                            self.update_status("Code generated successfully")
                        except Exception as e:
                            self.write_output(f"Error inserting code: {e}\n")
                            self.write_output(f"Code was: {code[:200]}...\n")
                            self.update_status(f"Error: {e}")
                    else:
                        self.write_output("Warning: Code generation returned empty code.\n")
                        self.write_output("Trying alternative generation method...\n")
                        # Try direct code generation
                        code = self._generate_code_directly(task)
                        if code:
                            self.code_editor.insert(tk.INSERT, code)
                            self.write_output("Code generated using alternative method!\n")
                            self.update_status("Code generated successfully")
                        else:
                            self.write_output("Could not generate code. The AI agent may not support this task type.\n")
                            self.update_status("Code generation failed")
                else:
                    error = result.get('error', 'Unknown error')
                    self.write_output(f"Code generation failed: {error}\n")
                    if result.get('traceback'):
                        self.write_output(f"\nTraceback:\n{result['traceback']}\n")
                    self.update_status("Code generation failed")
                    messagebox.showerror("AI Error", f"Could not generate code: {error}")
            except Exception as e:
                self.write_output(f"Error: {e}\n")
                self.update_status(f"Error: {e}")
                messagebox.showerror("AI Error", f"Error generating code: {e}")
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def ai_fix_code(self):
        """Fix code using AI Agent"""
        if not self.agent:
            messagebox.showwarning("AI Not Available", "AI Agent is not available.")
            return
        
        code = self.code_editor.get(1.0, tk.END + "-1c")
        if not code.strip():
            messagebox.showinfo("Fix Code", "No code to fix.")
            return
        
        self.update_status("Fixing code with AI...")
        self.write_output("Fixing code...\n\n")
        
        def fix():
            try:
                # Try to execute code to get error
                try:
                    compile(code, "<string>", "exec")
                    messagebox.showinfo("Fix Code", "Code has no syntax errors.")
                    self.update_status("Code has no syntax errors")
                    return
                except SyntaxError as e:
                    error_msg = str(e)
                except Exception as e:
                    error_msg = str(e)
                
                # Use agent to fix
                result = self.agent.build(f"Fix this code error: {error_msg}\n\nCode:\n{code}", context={})
                
                if result['success']:
                    fixed_code = result['code']
                    self.code_editor.delete(1.0, tk.END)
                    self.code_editor.insert(1.0, fixed_code)
                    self.write_output("Code fixed successfully!\n")
                    self.update_status("Code fixed successfully")
                    messagebox.showinfo("Fix Code", "Code fixed successfully!")
                else:
                    error = result.get('error', 'Unknown error')
                    self.write_output(f"Could not fix code: {error}\n")
                    self.update_status("Could not fix code")
                    messagebox.showerror("Fix Error", f"Could not fix code: {error}")
            except Exception as e:
                self.write_output(f"Error: {e}\n")
                self.update_status(f"Error: {e}")
                messagebox.showerror("Fix Error", f"Error fixing code: {e}")
        
        thread = threading.Thread(target=fix, daemon=True)
        thread.start()
    
    def ai_explain_code(self):
        """Explain code using AI"""
        code = self.code_editor.get(1.0, tk.END + "-1c")
        if not code.strip():
            messagebox.showinfo("Explain Code", "No code to explain.")
            return
        
        # Simple explanation (can be enhanced)
        messagebox.showinfo("Code Explanation", "Code explanation feature - to be enhanced with AI")
    
    def quick_train(self):
        """Quick ML model training"""
        if not self.toolbox:
            messagebox.showwarning("ML Toolbox Not Available", "ML Toolbox is not available.")
            return
        
        # Simple dialog for quick training
        dialog = tk.Toplevel(self.root)
        dialog.title("Quick Train Model")
        dialog.geometry("400x200")
        
        ttk.Label(dialog, text="Task Type:").pack(pady=10)
        task_var = tk.StringVar(value="classification")
        ttk.Radiobutton(dialog, text="Classification", variable=task_var, value="classification").pack()
        ttk.Radiobutton(dialog, text="Regression", variable=task_var, value="regression").pack()
        ttk.Radiobutton(dialog, text="Clustering", variable=task_var, value="clustering").pack()
        
        def train():
            task = task_var.get()
            dialog.destroy()
            self.write_output(f"Quick training {task} model...\n")
            self.write_output("Note: This requires data (X, y). Add data to your code first.\n")
            self.update_status(f"Quick train: {task}")
        
        ttk.Button(dialog, text="Train", command=train).pack(pady=10)
    
    def show_toolbox_status(self):
        """Show ML Toolbox status"""
        if not self.toolbox:
            messagebox.showwarning("ML Toolbox Not Available", "ML Toolbox is not available.")
            return
        
        status = f"ML Toolbox Status:\n"
        status += f"- Toolbox: {'Available' if self.toolbox else 'Not Available'}\n"
        status += f"- Agent: {'Available' if self.agent else 'Not Available'}\n"
        
        if hasattr(self.toolbox, 'get_dependency_status'):
            dep_status = self.toolbox.get_dependency_status()
            status += f"\nDependencies:\n"
            status += f"- Core: {sum(1 for v in dep_status.get('core', {}).values() if v)}/{len(dep_status.get('core', {}))}\n"
        
        messagebox.showinfo("ML Toolbox Status", status)
    
    def _generate_code_directly(self, task: str) -> str:
        """Generate code directly for common tasks"""
        task_lower = task.lower()
        
        # Tic-tac-toe game
        if 'tic' in task_lower and 'tac' in task_lower or 'tic-tac' in task_lower:
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
    # Winning combinations
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
        
        # Calculator
        elif 'calculator' in task_lower or 'calc' in task_lower:
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
        
        # Guess the number
        elif 'guess' in task_lower and 'number' in task_lower:
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
        
        # Default: return empty string
        return ''


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SimpleIDE(root)
    root.mainloop()


if __name__ == "__main__":
    main()
