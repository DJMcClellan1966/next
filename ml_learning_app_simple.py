"""
ML Learning App - Simple Desktop Version (No Flask Required)
Uses tkinter (built into Python) for Windows 11
"""
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading

sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml_toolbox import MLToolbox
    from ml_toolbox.ai_agent import MLCodeAgent
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("ML Toolbox not fully available, using simplified mode")

class MLLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Learning App - Learn Machine Learning")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize ML Toolbox
        try:
            self.toolbox = MLToolbox(check_dependencies=False) if TOOLBOX_AVAILABLE else None
            self.ai_agent = MLCodeAgent(use_llm=False) if TOOLBOX_AVAILABLE else None
        except:
            self.toolbox = None
            self.ai_agent = None
        
        # Student progress
        self.progress = {
            'modules_completed': 0,
            'exercises_solved': 0,
            'average_score': 0,
            'time_spent': 0
        }
        
        self.create_ui()
    
    def create_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#667eea', height=80)
        header.pack(fill=tk.X)
        
        title = tk.Label(header, text="🚀 ML Learning App", 
                        font=('Segoe UI', 24, 'bold'), 
                        bg='#667eea', fg='white')
        title.pack(pady=20)
        
        subtitle = tk.Label(header, text="Learn Machine Learning from the Ground Up", 
                           font=('Segoe UI', 12), 
                           bg='#667eea', fg='white')
        subtitle.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left sidebar
        sidebar = tk.Frame(main_container, bg='white', width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Navigation buttons
        nav_buttons = [
            ("📚 Modules", self.show_modules),
            ("💻 Exercises", self.show_exercises),
            ("📝 Quiz", self.show_quiz),
            ("📊 Progress", self.show_progress),
            ("🤖 AI Tutor", self.show_ai_tutor)
        ]
        
        self.current_section = None
        for text, command in nav_buttons:
            btn = tk.Button(sidebar, text=text, command=command,
                           font=('Segoe UI', 11), width=20, height=2,
                           bg='#667eea', fg='white', relief=tk.FLAT,
                           cursor='hand2')
            btn.pack(pady=5, padx=10, fill=tk.X)
        
        # Content area
        self.content_frame = tk.Frame(main_container, bg='white')
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Show modules by default
        self.show_modules()
    
    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def show_modules(self):
        self.clear_content()
        
        title = tk.Label(self.content_frame, text="Course Modules", 
                        font=('Segoe UI', 20, 'bold'), bg='white')
        title.pack(pady=20)
        
        modules = [
            {
                'title': 'Introduction to Machine Learning',
                'lessons': [
                    'What is Machine Learning?',
                    'Types of ML',
                    'ML Workflow',
                    'Your First Model'
                ]
            },
            {
                'title': 'Data Preprocessing',
                'lessons': [
                    'Data Cleaning',
                    'Feature Engineering',
                    'Data Transformation',
                    'Handling Missing Values'
                ]
            },
            {
                'title': 'Supervised Learning',
                'lessons': [
                    'Classification',
                    'Regression',
                    'Model Evaluation',
                    'Hyperparameter Tuning'
                ]
            }
        ]
        
        for module in modules:
            module_frame = tk.Frame(self.content_frame, bg='#f8f9fa', 
                                   relief=tk.RAISED, borderwidth=2)
            module_frame.pack(fill=tk.X, pady=10, padx=20)
            
            module_title = tk.Label(module_frame, text=module['title'],
                                   font=('Segoe UI', 14, 'bold'),
                                   bg='#f8f9fa', anchor='w')
            module_title.pack(fill=tk.X, padx=15, pady=10)
            
            for lesson in module['lessons']:
                lesson_btn = tk.Button(module_frame, text=f"📖 {lesson}",
                                      font=('Segoe UI', 10),
                                      bg='white', fg='#667eea',
                                      relief=tk.FLAT, anchor='w',
                                      command=lambda l=lesson: self.show_lesson(l),
                                      cursor='hand2')
                lesson_btn.pack(fill=tk.X, padx=15, pady=2)
    
    def show_lesson(self, lesson_title):
        content = f"""
Lesson: {lesson_title}

This is a comprehensive lesson about {lesson_title}.

Key Concepts:
• Fundamental principles
• Practical examples
• Code demonstrations
• Real-world applications

Let's learn together!
        """
        messagebox.showinfo(f"Lesson: {lesson_title}", content)
    
    def show_exercises(self):
        self.clear_content()
        
        title = tk.Label(self.content_frame, text="Interactive Exercises", 
                        font=('Segoe UI', 20, 'bold'), bg='white')
        title.pack(pady=20)
        
        exercises = [
            {
                'title': 'Train Your First Model',
                'instructions': 'Complete the code to train a classification model',
                'code': '''from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 1, 0, 1])

# TODO: Train a model
# result = toolbox.fit(???, ???, task_type='???')

# TODO: Make a prediction
# prediction = toolbox.predict(result['model'], [[5, 6]])
'''
            },
            {
                'title': 'Clean Your Data',
                'instructions': 'Use data cleaning utilities to clean missing values',
                'code': '''from ml_toolbox.compartment1_data.data_cleaning_utilities import DataCleaningUtilities
import numpy as np

# Create data with missing values
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

# TODO: Clean the data
# cleaner = ???
# cleaned = cleaner.clean_missing_values(???, strategy='???')
'''
            }
        ]
        
        for i, ex in enumerate(exercises):
            ex_frame = tk.Frame(self.content_frame, bg='#f8f9fa',
                               relief=tk.RAISED, borderwidth=2)
            ex_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)
            
            ex_title = tk.Label(ex_frame, text=ex['title'],
                               font=('Segoe UI', 14, 'bold'),
                               bg='#f8f9fa')
            ex_title.pack(pady=10)
            
            ex_instructions = tk.Label(ex_frame, text=ex['instructions'],
                                     font=('Segoe UI', 10),
                                     bg='#f8f9fa', wraplength=800)
            ex_instructions.pack(pady=5)
            
            code_text = scrolledtext.ScrolledText(ex_frame, height=10,
                                                  font=('Courier New', 10),
                                                  bg='#1e1e1e', fg='#d4d4d4',
                                                  insertbackground='white')
            code_text.insert('1.0', ex['code'])
            code_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            btn_frame = tk.Frame(ex_frame, bg='#f8f9fa')
            btn_frame.pack(pady=10)
            
            run_btn = tk.Button(btn_frame, text="▶ Run Code",
                               command=lambda c=code_text, idx=i: self.run_exercise(c, idx),
                               bg='#667eea', fg='white', font=('Segoe UI', 10),
                               padx=20, pady=5, cursor='hand2')
            run_btn.pack(side=tk.LEFT, padx=5)
            
            submit_btn = tk.Button(btn_frame, text="✓ Submit",
                                  command=lambda c=code_text, idx=i: self.submit_exercise(c, idx),
                                  bg='#28a745', fg='white', font=('Segoe UI', 10),
                                  padx=20, pady=5, cursor='hand2')
            submit_btn.pack(side=tk.LEFT, padx=5)
            
            result_label = tk.Label(ex_frame, text="", font=('Segoe UI', 10),
                                   bg='#f8f9fa', wraplength=800)
            result_label.pack(pady=5)
            code_text.result_label = result_label
    
    def run_exercise(self, code_widget, ex_id):
        code = code_widget.get('1.0', tk.END)
        
        def run_in_thread():
            try:
                exec_globals = {'__builtins__': __builtins__}
                if self.toolbox:
                    exec_globals['MLToolbox'] = MLToolbox
                    exec_globals['toolbox'] = self.toolbox
                    import numpy as np
                    exec_globals['np'] = np
                    exec_globals['numpy'] = np
                
                exec(code, exec_globals)
                code_widget.result_label.config(
                    text="✓ Code executed successfully!",
                    fg='green'
                )
            except Exception as e:
                code_widget.result_label.config(
                    text=f"✗ Error: {str(e)}",
                    fg='red'
                )
        
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def submit_exercise(self, code_widget, ex_id):
        code = code_widget.get('1.0', tk.END)
        
        # Simple validation
        is_correct = ('fit' in code or 'MLToolbox' in code) and 'predict' in code
        
        if is_correct:
            self.progress['exercises_solved'] += 1
            messagebox.showinfo("Success!", "🎉 Exercise completed correctly!\n\nGreat job!")
            code_widget.result_label.config(
                text="🎉 Correct! Great job!",
                fg='green'
            )
        else:
            messagebox.showwarning("Keep Trying", 
                                 "Not quite right. Make sure you're using:\n"
                                 "• toolbox.fit() to train\n"
                                 "• toolbox.predict() to predict")
            code_widget.result_label.config(
                text="Keep trying! Use toolbox.fit() and toolbox.predict()",
                fg='orange'
            )
    
    def show_quiz(self):
        self.clear_content()
        
        title = tk.Label(self.content_frame, text="ML Quiz", 
                        font=('Segoe UI', 20, 'bold'), bg='white')
        title.pack(pady=20)
        
        questions = [
            {
                'q': 'What is Machine Learning?',
                'options': [
                    'A type of programming language',
                    'A subset of AI that enables computers to learn from data',
                    'A database system',
                    'A web framework'
                ],
                'correct': 1
            },
            {
                'q': 'What are the main types of ML?',
                'options': [
                    'Supervised, Unsupervised, Reinforcement',
                    'Python, Java, C++',
                    'Classification, Regression, Clustering',
                    'Training, Testing, Validation'
                ],
                'correct': 0
            },
            {
                'q': 'What does toolbox.fit() do?',
                'options': [
                    'Fits data into a database',
                    'Trains a machine learning model',
                    'Fixes code errors',
                    'Creates a dashboard'
                ],
                'correct': 1
            }
        ]
        
        self.quiz_answers = {}
        
        for i, q in enumerate(questions):
            q_frame = tk.Frame(self.content_frame, bg='#f8f9fa',
                             relief=tk.RAISED, borderwidth=2)
            q_frame.pack(fill=tk.X, pady=10, padx=20)
            
            q_label = tk.Label(q_frame, text=f"{i+1}. {q['q']}",
                              font=('Segoe UI', 12, 'bold'),
                              bg='#f8f9fa', anchor='w')
            q_label.pack(fill=tk.X, padx=15, pady=10)
            
            var = tk.IntVar()
            for j, option in enumerate(q['options']):
                rb = tk.Radiobutton(q_frame, text=option, variable=var,
                                   value=j, font=('Segoe UI', 10),
                                   bg='#f8f9fa', anchor='w')
                rb.pack(fill=tk.X, padx=25, pady=2)
            self.quiz_answers[i] = (var, q['correct'])
        
        submit_btn = tk.Button(self.content_frame, text="Submit Quiz",
                               command=self.submit_quiz,
                               bg='#667eea', fg='white',
                               font=('Segoe UI', 12, 'bold'),
                               padx=30, pady=10, cursor='hand2')
        submit_btn.pack(pady=20)
    
    def submit_quiz(self):
        score = 0
        total = len(self.quiz_answers)
        
        for i, (var, correct) in self.quiz_answers.items():
            if var.get() == correct:
                score += 1
        
        percentage = (score / total * 100) if total > 0 else 0
        self.progress['average_score'] = int(percentage)
        
        messagebox.showinfo("Quiz Results",
                          f"Your Score: {score}/{total} ({percentage:.0f}%)\n\n"
                          f"{'Excellent!' if percentage >= 80 else 'Good job!' if percentage >= 60 else 'Keep studying!'}")
    
    def show_progress(self):
        self.clear_content()
        
        title = tk.Label(self.content_frame, text="Your Progress", 
                        font=('Segoe UI', 20, 'bold'), bg='white')
        title.pack(pady=20)
        
        # Progress card
        progress_frame = tk.Frame(self.content_frame, bg='#667eea',
                                 relief=tk.RAISED, borderwidth=2)
        progress_frame.pack(fill=tk.X, pady=10, padx=20)
        
        progress_label = tk.Label(progress_frame, text="Course Progress",
                                  font=('Segoe UI', 14, 'bold'),
                                  bg='#667eea', fg='white')
        progress_label.pack(pady=10)
        
        progress = (self.progress['modules_completed'] * 33 + 
                     self.progress['exercises_solved'] * 5)
        progress = min(100, progress)
        
        progress_bar = ttk.Progressbar(progress_frame, length=600,
                                       mode='determinate', maximum=100)
        progress_bar['value'] = progress
        progress_bar.pack(pady=10, padx=20)
        
        progress_text = tk.Label(progress_frame, text=f"{progress}% Complete",
                                font=('Segoe UI', 12),
                                bg='#667eea', fg='white')
        progress_text.pack(pady=5)
        
        # Metrics
        metrics_frame = tk.Frame(self.content_frame, bg='white')
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=20, padx=20)
        
        metrics = [
            ('Modules Completed', self.progress['modules_completed']),
            ('Exercises Solved', self.progress['exercises_solved']),
            ('Average Score', f"{self.progress['average_score']}%"),
            ('Time Spent', f"{self.progress['time_spent']} hours")
        ]
        
        for i, (label, value) in enumerate(metrics):
            metric_frame = tk.Frame(metrics_frame, bg='white',
                                   relief=tk.RAISED, borderwidth=1)
            metric_frame.grid(row=i//2, column=i%2, padx=10, pady=10,
                            sticky='nsew', ipadx=20, ipady=20)
            
            value_label = tk.Label(metric_frame, text=str(value),
                                  font=('Segoe UI', 24, 'bold'),
                                  bg='white', fg='#667eea')
            value_label.pack()
            
            name_label = tk.Label(metric_frame, text=label,
                                 font=('Segoe UI', 10),
                                 bg='white', fg='#666')
            name_label.pack()
        
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
    
    def show_ai_tutor(self):
        self.clear_content()
        
        title = tk.Label(self.content_frame, text="AI Tutor", 
                        font=('Segoe UI', 20, 'bold'), bg='white')
        title.pack(pady=20)
        
        tutor_frame = tk.Frame(self.content_frame, bg='#f8f9fa',
                              relief=tk.RAISED, borderwidth=2)
        tutor_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)
        
        question_label = tk.Label(tutor_frame, text="Ask a Question:",
                                 font=('Segoe UI', 12, 'bold'),
                                 bg='#f8f9fa', anchor='w')
        question_label.pack(fill=tk.X, padx=15, pady=10)
        
        question_text = scrolledtext.ScrolledText(tutor_frame, height=5,
                                                  font=('Segoe UI', 11),
                                                  wrap=tk.WORD)
        question_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        question_text.insert('1.0', "What is machine learning?")
        
        ask_btn = tk.Button(tutor_frame, text="Ask AI Tutor",
                           command=lambda: self.ask_tutor(question_text),
                           bg='#667eea', fg='white',
                           font=('Segoe UI', 12, 'bold'),
                           padx=30, pady=10, cursor='hand2')
        ask_btn.pack(pady=15)
        
        response_label = tk.Label(tutor_frame, text="Response:",
                                 font=('Segoe UI', 12, 'bold'),
                                 bg='#f8f9fa', anchor='w')
        response_label.pack(fill=tk.X, padx=15, pady=(20, 5))
        
        response_text = scrolledtext.ScrolledText(tutor_frame, height=10,
                                                 font=('Segoe UI', 11),
                                                 wrap=tk.WORD, state=tk.DISABLED)
        response_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        question_text.response_text = response_text
    
    def ask_tutor(self, question_widget):
        question = question_widget.get('1.0', tk.END).strip()
        
        def get_answer():
            if self.ai_agent:
                try:
                    result = self.ai_agent.build(f"Explain simply: {question}")
                    answer = result.get('code', f"Great question! {question} is important in ML...")
                except:
                    answer = f"Great question about '{question}'! In machine learning..."
            else:
                answer = f"""Great question! "{question}" is a fundamental concept in machine learning.

Here's a simple explanation:

Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

Key points:
• ML learns patterns from data
• It improves with more data
• It can make predictions
• It's used in many applications

Would you like to know more about any specific aspect?"""
            
            question_widget.response_text.config(state=tk.NORMAL)
            question_widget.response_text.delete('1.0', tk.END)
            question_widget.response_text.insert('1.0', answer)
            question_widget.response_text.config(state=tk.DISABLED)
        
        threading.Thread(target=get_answer, daemon=True).start()


if __name__ == '__main__':
    print("="*60)
    print("🚀 ML Learning App - Desktop Version")
    print("="*60)
    print("\nStarting application...")
    print("="*60)
    
    root = tk.Tk()
    app = MLLearningApp(root)
    root.mainloop()
