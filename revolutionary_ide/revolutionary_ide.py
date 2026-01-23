"""
Revolutionary IDE - Built with ML Toolbox & AI Agent
A complete IDE that uses AI to help you build ML code
"""
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import json
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_toolbox import MLToolbox
    from ml_toolbox.ai_agent import MLCodeAgent
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("Warning: ML Toolbox not available")


class RevolutionaryIDE:
    """
    Revolutionary IDE built with ML Toolbox & AI Agent
    
    Features:
    - AI-powered code generation
    - ML Toolbox integration
    - Real-time error fixing
    - Visual model training
    - Interactive ML pipeline builder
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Revolutionary IDE - ML Toolbox & AI Agent")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.toolbox = MLToolbox() if TOOLBOX_AVAILABLE else None
        self.agent = MLCodeAgent(use_llm=False, use_pattern_composition=True) if TOOLBOX_AVAILABLE else None
        
        # Code execution queue
        self.execution_queue = queue.Queue()
        
        # Create UI
        self.create_ui()
        
        # Start execution thread
        self.start_execution_thread()
    
    def create_ui(self):
        """Create the IDE UI"""
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Code editor and AI
        left_panel = ttk.PanedWindow(main_container, orient=tk.VERTICAL)
        main_container.add(left_panel, weight=2)
        
        # Code editor
        editor_frame = ttk.Frame(left_panel)
        left_panel.add(editor_frame, weight=2)
        
        ttk.Label(editor_frame, text="Code Editor", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        
        self.code_editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#ffffff"
        )
        self.code_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Editor toolbar
        editor_toolbar = ttk.Frame(editor_frame)
        editor_toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(editor_toolbar, text="Run Code", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(editor_toolbar, text="AI Fix Error", command=self.ai_fix_error).pack(side=tk.LEFT, padx=2)
        ttk.Button(editor_toolbar, text="Clear", command=self.clear_editor).pack(side=tk.LEFT, padx=2)
        
        # AI Assistant panel
        ai_frame = ttk.Frame(left_panel)
        left_panel.add(ai_frame, weight=1)
        
        ttk.Label(ai_frame, text="AI Assistant", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        
        # AI input
        ai_input_frame = ttk.Frame(ai_frame)
        ai_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ai_input_frame, text="Describe what you want:").pack(side=tk.LEFT, padx=5)
        self.ai_input = ttk.Entry(ai_input_frame, font=("Arial", 10))
        self.ai_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.ai_input.bind("<Return>", lambda e: self.ai_generate_code())
        
        ttk.Button(ai_input_frame, text="Generate", command=self.ai_generate_code).pack(side=tk.LEFT, padx=5)
        
        # AI output
        self.ai_output = scrolledtext.ScrolledText(
            ai_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            height=8,
            bg="#252526",
            fg="#cccccc"
        )
        self.ai_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - ML Toolbox and Results
        right_panel = ttk.PanedWindow(main_container, orient=tk.VERTICAL)
        main_container.add(right_panel, weight=1)
        
        # ML Toolbox panel
        toolbox_frame = ttk.LabelFrame(right_panel, text="ML Toolbox", padding=10)
        right_panel.add(toolbox_frame, weight=1)
        
        # Quick train
        train_frame = ttk.Frame(toolbox_frame)
        train_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(train_frame, text="Quick Train Model:").pack(side=tk.LEFT, padx=5)
        self.task_type = ttk.Combobox(train_frame, values=["classification", "regression", "clustering"], width=15)
        self.task_type.set("classification")
        self.task_type.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(train_frame, text="Train", command=self.quick_train).pack(side=tk.LEFT, padx=5)
        
        # Model info
        self.model_info = scrolledtext.ScrolledText(
            toolbox_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=10,
            bg="#1e1e1e",
            fg="#d4d4d4"
        )
        self.model_info.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Results panel
        results_frame = ttk.LabelFrame(right_panel, text="Results", padding=10)
        right_panel.add(results_frame, weight=1)
        
        self.results_output = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#00ff00"
        )
        self.results_output.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Menu bar
        self.create_menu()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="AI Generate Code", command=self.ai_generate_code)
        edit_menu.add_command(label="AI Fix Error", command=self.ai_fix_error)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Train Model", command=self.quick_train)
        tools_menu.add_command(label="Show Models", command=self.show_models)
    
    def ai_generate_code(self):
        """Generate code using AI Agent"""
        if not self.agent:
            messagebox.showerror("Error", "AI Agent not available")
            return
        
        prompt = self.ai_input.get()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a task description")
            return
        
        self.status_bar.config(text="Generating code...")
        self.ai_output.insert(tk.END, f"Generating code for: {prompt}\n")
        self.ai_output.see(tk.END)
        
        def generate():
            try:
                result = self.agent.build(prompt)
                
                if result.get('success'):
                    code = result['code']
                    self.execution_queue.put(('insert_code', code))
                    self.execution_queue.put(('ai_output', f"Code generated successfully!\n\n{code}\n\n"))
                else:
                    error = result.get('error', 'Unknown error')
                    self.execution_queue.put(('ai_output', f"Generation failed: {error}\n\n"))
            except Exception as e:
                self.execution_queue.put(('ai_output', f"Error: {str(e)}\n\n"))
            finally:
                self.execution_queue.put(('status', 'Ready'))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def ai_fix_error(self):
        """Fix code error using AI Agent"""
        if not self.agent:
            messagebox.showerror("Error", "AI Agent not available")
            return
        
        code = self.code_editor.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to fix")
            return
        
        self.status_bar.config(text="Fixing error...")
        self.ai_output.insert(tk.END, "Fixing code error...\n")
        self.ai_output.see(tk.END)
        
        def fix():
            try:
                result = self.agent.build(f"Fix this code: {code}")
                
                if result.get('success'):
                    fixed_code = result['code']
                    self.execution_queue.put(('replace_code', fixed_code))
                    self.execution_queue.put(('ai_output', f"Code fixed!\n\n{fixed_code}\n\n"))
                else:
                    error = result.get('error', 'Unknown error')
                    self.execution_queue.put(('ai_output', f"Fix failed: {error}\n\n"))
            except Exception as e:
                self.execution_queue.put(('ai_output', f"Error: {str(e)}\n\n"))
            finally:
                self.execution_queue.put(('status', 'Ready'))
        
        threading.Thread(target=fix, daemon=True).start()
    
    def quick_train(self):
        """Quick train model"""
        if not self.toolbox:
            messagebox.showerror("Error", "ML Toolbox not available")
            return
        
        self.status_bar.config(text="Training model...")
        self.model_info.insert(tk.END, "Training model...\n")
        self.model_info.see(tk.END)
        
        def train():
            try:
                import numpy as np
                
                # Generate sample data
                X = np.random.randn(200, 10)
                task_type = self.task_type.get()
                
                if task_type == 'classification':
                    y = np.random.randint(0, 2, 200)
                elif task_type == 'regression':
                    y = np.random.randn(200)
                else:  # clustering
                    y = None
                
                # Train model
                result = self.toolbox.fit(X, y, task_type=task_type)
                
                # Display results
                info = f"Model trained successfully!\n"
                info += f"Task: {task_type}\n"
                if 'accuracy' in result:
                    info += f"Accuracy: {result['accuracy']:.2%}\n"
                if 'r2_score' in result:
                    info += f"R² Score: {result['r2_score']:.4f}\n"
                info += f"\nModel ID: {result.get('model_id', 'N/A')}\n"
                
                self.execution_queue.put(('model_info', info))
                self.execution_queue.put(('status', 'Model trained successfully'))
            except Exception as e:
                self.execution_queue.put(('model_info', f"Error: {str(e)}\n"))
                self.execution_queue.put(('status', 'Training failed'))
        
        threading.Thread(target=train, daemon=True).start()
    
    def run_code(self):
        """Run code in editor"""
        code = self.code_editor.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to run")
            return
        
        self.status_bar.config(text="Running code...")
        self.results_output.delete("1.0", tk.END)
        self.results_output.insert(tk.END, "Running code...\n\n")
        
        def execute():
            try:
                # Create execution namespace
                namespace = {
                    '__builtins__': __builtins__,
                    'MLToolbox': self.toolbox.__class__ if self.toolbox else None,
                    'toolbox': self.toolbox,
                    'agent': self.agent
                }
                
                # Execute code
                exec(code, namespace)
                
                self.execution_queue.put(('results', "Code executed successfully!\n"))
                self.execution_queue.put(('status', 'Code executed'))
            except Exception as e:
                error_msg = f"Error: {str(e)}\n"
                self.execution_queue.put(('results', error_msg))
                self.execution_queue.put(('status', 'Execution failed'))
        
        threading.Thread(target=execute, daemon=True).start()
    
    def start_execution_thread(self):
        """Start thread to process execution queue"""
        def process_queue():
            while True:
                try:
                    item = self.execution_queue.get(timeout=0.1)
                    action, data = item
                    
                    if action == 'insert_code':
                        self.code_editor.insert(tk.END, data + "\n\n")
                    elif action == 'replace_code':
                        self.code_editor.delete("1.0", tk.END)
                        self.code_editor.insert("1.0", data)
                    elif action == 'ai_output':
                        self.ai_output.insert(tk.END, data)
                        self.ai_output.see(tk.END)
                    elif action == 'model_info':
                        self.model_info.delete("1.0", tk.END)
                        self.model_info.insert("1.0", data)
                    elif action == 'results':
                        self.results_output.insert(tk.END, data)
                        self.results_output.see(tk.END)
                    elif action == 'status':
                        self.status_bar.config(text=data)
                    
                    self.execution_queue.task_done()
                except queue.Empty:
                    continue
        
        threading.Thread(target=process_queue, daemon=True).start()
    
    def clear_editor(self):
        """Clear code editor"""
        self.code_editor.delete("1.0", tk.END)
    
    def new_file(self):
        """Create new file"""
        self.code_editor.delete("1.0", tk.END)
        self.status_bar.config(text="New file")
    
    def open_file(self):
        """Open file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'r') as f:
                content = f.read()
                self.code_editor.delete("1.0", tk.END)
                self.code_editor.insert("1.0", content)
            self.status_bar.config(text=f"Opened: {filename}")
    
    def save_file(self):
        """Save file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filename:
            content = self.code_editor.get("1.0", tk.END)
            with open(filename, 'w') as f:
                f.write(content)
            self.status_bar.config(text=f"Saved: {filename}")
    
    def show_models(self):
        """Show registered models"""
        if not self.toolbox or not hasattr(self.toolbox, 'model_registry'):
            messagebox.showinfo("Info", "Model registry not available")
            return
        
        registry = self.toolbox.model_registry
        if registry:
            models = registry.list_models()
            info = "Registered Models:\n\n"
            for model in models:
                info += f"Name: {model.get('name', 'N/A')}\n"
                info += f"Version: {model.get('version', 'N/A')}\n"
                info += f"Stage: {model.get('stage', 'N/A')}\n\n"
            
            self.model_info.delete("1.0", tk.END)
            self.model_info.insert("1.0", info)


def main():
    """Launch Revolutionary IDE"""
    root = tk.Tk()
    app = RevolutionaryIDE(root)
    root.mainloop()


if __name__ == '__main__':
    main()
