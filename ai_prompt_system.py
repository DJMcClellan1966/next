"""
AI Prompt System for Non-Technical Users
Natural language interface for ML Toolbox

Features:
- Natural language understanding
- Question-based prompts
- Data processing automation
- Human-readable reports
- Guided workflows
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import warnings
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    warnings.warn("ML Toolbox not available")


class AIPromptSystem:
    """
    AI Prompt System for Non-Technical Users
    
    Natural language interface to ML Toolbox
    """
    
    def __init__(self):
        """Initialize AI Prompt System"""
        self.toolbox = MLToolbox() if TOOLBOX_AVAILABLE else None
        self.conversation_history = []
        self.user_data = {}
        self.current_task = None
    
    def start_conversation(self) -> str:
        """
        Start conversation with user
        
        Returns:
            Welcome message and first question
        """
        welcome = """
🤖 Welcome to the ML Toolbox AI Assistant!

I can help you analyze your data and create machine learning models - no coding required!

Let me ask you a few questions to understand what you need:
"""
        return welcome + self._ask_about_task()
    
    def _ask_about_task(self) -> str:
        """Ask about user's task"""
        return """
📋 What would you like to do today?

1. Analyze data and find patterns
2. Predict future values or outcomes
3. Group similar items together
4. Understand what factors are important
5. Something else (describe it)

Please type the number or describe what you need:"""
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and determine next steps
        
        Args:
            user_input: User's natural language input
            
        Returns:
            Response with next question or action
        """
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Determine intent
        intent = self._understand_intent(user_input)
        
        # Process based on intent
        if intent['type'] == 'task_selection':
            return self._handle_task_selection(intent)
        elif intent['type'] == 'data_upload':
            return self._handle_data_upload(intent)
        elif intent['type'] == 'question':
            return self._handle_question(intent)
        elif intent['type'] == 'execute':
            return self._execute_task(intent)
        else:
            return {
                'response': "I'm not sure I understand. Could you rephrase that?",
                'next_question': self._ask_about_task(),
                'status': 'clarification_needed'
            }
    
    def _understand_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Understand user intent from natural language
        
        Args:
            user_input: User's input
            
        Returns:
            Intent dictionary
        """
        input_lower = user_input.lower()
        
        # Task selection
        if any(word in input_lower for word in ['1', 'analyze', 'pattern', 'find']):
            return {'type': 'task_selection', 'task': 'analysis'}
        elif any(word in input_lower for word in ['2', 'predict', 'forecast', 'future']):
            return {'type': 'task_selection', 'task': 'prediction'}
        elif any(word in input_lower for word in ['3', 'group', 'cluster', 'similar']):
            return {'type': 'task_selection', 'task': 'clustering'}
        elif any(word in input_lower for word in ['4', 'important', 'factor', 'feature']):
            return {'type': 'task_selection', 'task': 'feature_importance'}
        elif any(word in input_lower for word in ['upload', 'data', 'file', 'csv', 'excel']):
            return {'type': 'data_upload', 'data': user_input}
        elif any(word in input_lower for word in ['yes', 'ok', 'sure', 'go ahead', 'run', 'execute']):
            return {'type': 'execute', 'action': 'run_analysis'}
        else:
            return {'type': 'question', 'content': user_input}
    
    def _handle_task_selection(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task selection"""
        task = intent.get('task')
        self.current_task = task
        
        task_descriptions = {
            'analysis': 'data analysis and pattern finding',
            'prediction': 'predicting future values or outcomes',
            'clustering': 'grouping similar items together',
            'feature_importance': 'understanding important factors'
        }
        
        response = f"""
✅ Great! I'll help you with {task_descriptions.get(task, 'your task')}.

Now I need to know about your data:
"""
        
        next_question = self._ask_about_data()
        
        return {
            'response': response,
            'next_question': next_question,
            'status': 'data_needed',
            'task': task
        }
    
    def _ask_about_data(self) -> str:
        """Ask about user's data"""
        return """
📊 Tell me about your data:

1. Do you have a data file ready? (CSV, Excel, etc.)
   → If yes, please provide the file path or describe where it is

2. What does your data contain?
   → Describe what columns/fields you have (e.g., "sales, date, region")

3. What are you trying to predict or analyze?
   → Describe your goal (e.g., "predict sales next month")

Please provide this information:"""
    
    def _handle_data_upload(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data upload"""
        return {
            'response': "I understand you want to upload data. Please provide the file path or describe your data structure.",
            'next_question': "What is the path to your data file, or can you describe what data you have?",
            'status': 'awaiting_data'
        }
    
    def _handle_question(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user questions"""
        content = intent.get('content', '')
        
        # Answer common questions
        if 'what can' in content.lower() or 'what does' in content.lower():
            return {
                'response': """
I can help you with:
- Analyzing data to find patterns and insights
- Building models to predict outcomes
- Grouping similar items together
- Understanding which factors are most important
- Creating reports with your findings

What would you like to do?""",
                'next_question': self._ask_about_task(),
                'status': 'question_answered'
            }
        else:
            return {
                'response': "I understand. Let me help you with that.",
                'next_question': self._ask_about_data(),
                'status': 'proceeding'
            }
    
    def _execute_task(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected task"""
        if not self.current_task:
            return {
                'response': "I need to know what task you want to perform first.",
                'next_question': self._ask_about_task(),
                'status': 'task_needed'
            }
        
        if 'data' not in self.user_data:
            return {
                'response': "I need your data first. Please provide your data file or describe your data.",
                'next_question': self._ask_about_data(),
                'status': 'data_needed'
            }
        
        # Execute task
        try:
            result = self._run_analysis()
            report = self._generate_human_readable_report(result)
            
            return {
                'response': "✅ Analysis complete! Here are your results:",
                'report': report,
                'status': 'completed',
                'next_question': "Would you like to:\n1. Save this report\n2. Try a different analysis\n3. Ask questions about the results"
            }
        except Exception as e:
            return {
                'response': f"I encountered an error: {str(e)}",
                'next_question': "Would you like to try again or provide different data?",
                'status': 'error'
            }
    
    def load_data(self, file_path: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Load user's data
        
        Args:
            file_path: Path to data file
            description: Optional description of data
            
        Returns:
            Data loading result
        """
        try:
            # Try to load as CSV
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                # Try CSV as default
                df = pd.read_csv(file_path)
            
            self.user_data['dataframe'] = df
            self.user_data['file_path'] = file_path
            self.user_data['description'] = description
            
            # Analyze data structure
            data_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            self.user_data['info'] = data_info
            
            return {
                'success': True,
                'message': f"✅ Data loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.",
                'data_info': data_info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Could not load data: {str(e)}"
            }
    
    def set_task_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set task parameters from user input
        
        Args:
            parameters: Task parameters
            
        Returns:
            Confirmation
        """
        self.user_data['parameters'] = parameters
        return {'success': True, 'message': "Parameters set successfully"}
    
    def _run_analysis(self) -> Dict[str, Any]:
        """Run analysis based on current task"""
        if not self.toolbox:
            raise ValueError("ML Toolbox not available")
        
        df = self.user_data.get('dataframe')
        if df is None:
            raise ValueError("No data loaded")
        
        task = self.current_task
        results = {}
        
        if task == 'analysis':
            results = self._run_data_analysis(df)
        elif task == 'prediction':
            results = self._run_prediction(df)
        elif task == 'clustering':
            results = self._run_clustering(df)
        elif task == 'feature_importance':
            results = self._run_feature_importance(df)
        
        return results
    
    def _run_data_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run data analysis"""
        results = {
            'task': 'data_analysis',
            'summary': {},
            'insights': []
        }
        
        # Basic statistics
        results['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        # Statistical summary
        numeric_cols = df.select_dtypes(include=[np.number])
        if len(numeric_cols.columns) > 0:
            results['statistics'] = numeric_cols.describe().to_dict()
        
        # Insights
        insights = []
        
        # Missing data
        missing = df.isnull().sum()
        if missing.sum() > 0:
            insights.append(f"⚠️ Found missing data in {missing[missing > 0].count()} columns")
        
        # Outliers (simplified)
        for col in numeric_cols.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                insights.append(f"📊 Found {len(outliers)} potential outliers in '{col}'")
        
        results['insights'] = insights
        
        return results
    
    def _run_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run prediction task"""
        # Ask user for target column if not specified
        target = self.user_data.get('parameters', {}).get('target_column')
        
        if not target:
            # Try to infer
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                target = numeric_cols[-1]  # Use last numeric column as default
        
        if target and target in df.columns:
            # Prepare data
            X = df.drop(columns=[target])
            y = df[target]
            
            # Use simple ML tasks
            simple_ml = self.toolbox.algorithms.get_simple_ml_tasks()
            
            # Determine if classification or regression
            if y.dtype in ['object', 'category'] or len(y.unique()) < 10:
                result = simple_ml.train_classifier(X.values, y.values, model_type='random_forest')
            else:
                result = simple_ml.train_regressor(X.values, y.values, model_type='random_forest')
            
            return {
                'task': 'prediction',
                'model_type': 'classification' if y.dtype in ['object', 'category'] or len(y.unique()) < 10 else 'regression',
                'target_column': target,
                'accuracy': result.get('accuracy', result.get('r2_score', 0)),
                'model': result.get('model')
            }
        else:
            return {
                'task': 'prediction',
                'error': 'Target column not specified or not found'
            }
    
    def _run_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run clustering analysis"""
        # Use numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {'error': 'No numeric columns found for clustering'}
        
        # Determine number of clusters
        n_clusters = self.user_data.get('parameters', {}).get('n_clusters', 3)
        
        # Use sklearn KMeans (simplified)
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(numeric_df.values)
            
            # Add cluster labels to dataframe
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = labels
            
            return {
                'task': 'clustering',
                'n_clusters': n_clusters,
                'cluster_labels': labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'data_with_clusters': df_with_clusters
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _run_feature_importance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run feature importance analysis"""
        target = self.user_data.get('parameters', {}).get('target_column')
        
        if not target:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                target = numeric_cols[-1]
        
        if target and target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]
            
            # Use random forest for feature importance
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if y.dtype in ['object', 'category'] or len(y.unique()) < 10:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(X.select_dtypes(include=[np.number]).values, y.values)
                
                # Get feature importance
                importances = model.feature_importances_
                feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
                
                importance_dict = dict(zip(feature_names, importances))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                return {
                    'task': 'feature_importance',
                    'target_column': target,
                    'feature_importance': dict(sorted_importance),
                    'top_features': [f[0] for f in sorted_importance[:5]]
                }
            except Exception as e:
                return {'error': str(e)}
        else:
            return {'error': 'Target column not specified'}
    
    def _generate_human_readable_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable report
        
        Args:
            results: Analysis results
            
        Returns:
            Human-readable report string
        """
        task = results.get('task', 'unknown')
        
        report = f"""
{'='*80}
📊 ANALYSIS REPORT
{'='*80}

Task: {task.replace('_', ' ').title()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if task == 'data_analysis':
            summary = results.get('summary', {})
            report += f"""
📈 DATA SUMMARY:
- Total Rows: {summary.get('total_rows', 'N/A')}
- Total Columns: {summary.get('total_columns', 'N/A')}
- Numeric Columns: {len(summary.get('numeric_columns', []))}
- Categorical Columns: {len(summary.get('categorical_columns', []))}

"""
            
            insights = results.get('insights', [])
            if insights:
                report += "💡 KEY INSIGHTS:\n"
                for insight in insights:
                    report += f"  • {insight}\n"
                report += "\n"
        
        elif task == 'prediction':
            report += f"""
🎯 PREDICTION MODEL RESULTS:

Model Type: {results.get('model_type', 'N/A').title()}
Target Column: {results.get('target_column', 'N/A')}
"""
            
            accuracy = results.get('accuracy')
            if accuracy:
                if results.get('model_type') == 'classification':
                    report += f"Accuracy: {accuracy:.2%}\n\n"
                else:
                    report += f"R² Score: {accuracy:.4f}\n\n"
            
            report += """
💡 INTERPRETATION:
- Higher accuracy/R² means better predictions
- The model learned patterns from your data
- You can use this model to predict new values

"""
        
        elif task == 'clustering':
            report += f"""
🔍 CLUSTERING RESULTS:

Number of Clusters: {results.get('n_clusters', 'N/A')}
Items Analyzed: {len(results.get('cluster_labels', []))}

"""
            
            # Cluster distribution
            labels = results.get('cluster_labels', [])
            if labels:
                from collections import Counter
                cluster_counts = Counter(labels)
                report += "📊 CLUSTER DISTRIBUTION:\n"
                for cluster, count in sorted(cluster_counts.items()):
                    report += f"  • Cluster {cluster}: {count} items ({count/len(labels)*100:.1f}%)\n"
                report += "\n"
            
            report += """
💡 INTERPRETATION:
- Items in the same cluster are similar
- Different clusters represent different groups
- Use clusters to understand patterns in your data

"""
        
        elif task == 'feature_importance':
            importance = results.get('feature_importance', {})
            top_features = results.get('top_features', [])
            
            report += f"""
🎯 FEATURE IMPORTANCE ANALYSIS:

Target Column: {results.get('target_column', 'N/A')}

TOP 5 MOST IMPORTANT FACTORS:
"""
            
            for i, feature in enumerate(top_features, 1):
                score = importance.get(feature, 0)
                report += f"  {i}. {feature}: {score:.4f} ({score*100:.2f}%)\n"
            
            report += """
💡 INTERPRETATION:
- Higher scores mean more important for predictions
- Focus on top factors for better results
- Less important factors can potentially be removed

"""
        
        report += f"""
{'='*80}
✅ Analysis Complete!
{'='*80}

Need help understanding these results? Just ask!
"""
        
        return report
    
    def ask_clarifying_question(self, context: str) -> str:
        """
        Ask clarifying question based on context
        
        Args:
            context: Current context
            
        Returns:
            Clarifying question
        """
        questions = {
            'data_needed': "What data do you have? Please describe your data file or provide the path.",
            'target_needed': "What are you trying to predict? Please name the column or describe what you want to predict.",
            'task_unclear': "I'm not sure what you need. Could you describe what you want to do with your data?",
            'parameters_needed': "I need a bit more information. Could you answer the questions above?"
        }
        
        return questions.get(context, "Could you provide more details?")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            'total_messages': len(self.conversation_history),
            'current_task': self.current_task,
            'data_loaded': 'dataframe' in self.user_data,
            'data_info': self.user_data.get('info', {}),
            'history': self.conversation_history[-10:]  # Last 10 messages
        }


class GuidedWorkflow:
    """
    Guided Workflow System
    
    Step-by-step guidance for non-technical users
    """
    
    def __init__(self, prompt_system: AIPromptSystem):
        """
        Args:
            prompt_system: AI Prompt System instance
        """
        self.prompt_system = prompt_system
        self.current_step = 0
        self.workflow_steps = []
    
    def create_workflow(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Create guided workflow for task
        
        Args:
            task_type: Type of task
            
        Returns:
            Workflow steps
        """
        workflows = {
            'analysis': [
                {
                    'step': 1,
                    'question': 'What is the path to your data file? (CSV, Excel, etc.)',
                    'help': 'Example: "C:/Users/MyData/data.csv" or "my_data.xlsx"',
                    'required': True
                },
                {
                    'step': 2,
                    'question': 'What would you like to analyze? (patterns, trends, relationships, etc.)',
                    'help': 'Example: "Find patterns in sales data" or "Understand customer behavior"',
                    'required': False
                },
                {
                    'step': 3,
                    'question': 'Any specific columns you want to focus on? (optional)',
                    'help': 'Example: "sales, date, region" or "all columns"',
                    'required': False
                }
            ],
            'prediction': [
                {
                    'step': 1,
                    'question': 'What is the path to your data file?',
                    'help': 'Example: "C:/Users/MyData/data.csv"',
                    'required': True
                },
                {
                    'step': 2,
                    'question': 'What do you want to predict? (name the column)',
                    'help': 'Example: "sales" or "customer_churn" or "price"',
                    'required': True
                },
                {
                    'step': 3,
                    'question': 'What type of prediction? (number/amount, category/class, or I\'m not sure)',
                    'help': 'Example: "number" for sales amount, "category" for yes/no',
                    'required': False
                }
            ],
            'clustering': [
                {
                    'step': 1,
                    'question': 'What is the path to your data file?',
                    'help': 'Example: "C:/Users/MyData/data.csv"',
                    'required': True
                },
                {
                    'step': 2,
                    'question': 'How many groups do you want? (or type "auto" to let me decide)',
                    'help': 'Example: "3" for 3 groups, or "auto"',
                    'required': False
                },
                {
                    'step': 3,
                    'question': 'Which columns should I use? (or "all" for all numeric columns)',
                    'help': 'Example: "age, income, spending" or "all"',
                    'required': False
                }
            ]
        }
        
        return workflows.get(task_type, [])
    
    def process_step_response(self, step: int, response: str) -> Dict[str, Any]:
        """
        Process response to workflow step
        
        Args:
            step: Step number
            response: User's response
            
        Returns:
            Processing result
        """
        # Store response
        if 'responses' not in self.prompt_system.user_data:
            self.prompt_system.user_data['responses'] = {}
        
        self.prompt_system.user_data['responses'][step] = response
        
        # Check if workflow complete
        workflow = self.workflow_steps
        if step >= len(workflow):
            # Execute task
            return {
                'status': 'complete',
                'message': 'All information collected. Running analysis...',
                'next_action': 'execute'
            }
        else:
            # Next step
            next_step = workflow[step]
            return {
                'status': 'continue',
                'next_step': next_step,
                'question': next_step['question'],
                'help': next_step.get('help', '')
            }


class ReportGenerator:
    """
    Report Generator
    
    Creates human-readable reports from analysis results
    """
    
    @staticmethod
    def generate_executive_summary(results: Dict[str, Any]) -> str:
        """
        Generate executive summary
        
        Args:
            results: Analysis results
            
        Returns:
            Executive summary
        """
        task = results.get('task', 'analysis')
        
        summary = f"""
EXECUTIVE SUMMARY
{'='*60}

Analysis Type: {task.replace('_', ' ').title()}
Date: {datetime.now().strftime('%B %d, %Y')}

"""
        
        if task == 'prediction':
            accuracy = results.get('accuracy', 0)
            summary += f"""
Key Finding: The prediction model achieved {accuracy:.1%} accuracy.

This means the model can reliably predict outcomes based on the patterns 
found in your data.
"""
        elif task == 'clustering':
            n_clusters = results.get('n_clusters', 0)
            summary += f"""
Key Finding: Your data naturally forms {n_clusters} distinct groups.

These groups represent different patterns or segments in your data that 
can help you make better decisions.
"""
        elif task == 'feature_importance':
            top_features = results.get('top_features', [])
            if top_features:
                summary += f"""
Key Finding: The most important factors are: {', '.join(top_features[:3])}

Focusing on these factors will have the greatest impact on your outcomes.
"""
        
        return summary
    
    @staticmethod
    def generate_detailed_report(results: Dict[str, Any], include_charts: bool = False) -> str:
        """
        Generate detailed report
        
        Args:
            results: Analysis results
            include_charts: Whether to include chart descriptions
            
        Returns:
            Detailed report
        """
        report = ReportGenerator.generate_executive_summary(results)
        
        # Add detailed sections
        task = results.get('task')
        
        if task == 'data_analysis':
            report += "\n" + ReportGenerator._generate_data_analysis_details(results)
        elif task == 'prediction':
            report += "\n" + ReportGenerator._generate_prediction_details(results)
        elif task == 'clustering':
            report += "\n" + ReportGenerator._generate_clustering_details(results)
        elif task == 'feature_importance':
            report += "\n" + ReportGenerator._generate_feature_importance_details(results)
        
        return report
    
    @staticmethod
    def _generate_data_analysis_details(results: Dict[str, Any]) -> str:
        """Generate data analysis details"""
        details = """
DETAILED FINDINGS
{'='*60}

"""
        summary = results.get('summary', {})
        details += f"Data Overview:\n"
        details += f"- Total Records: {summary.get('total_rows', 'N/A'):,}\n"
        details += f"- Data Fields: {summary.get('total_columns', 'N/A')}\n\n"
        
        insights = results.get('insights', [])
        if insights:
            details += "Key Observations:\n"
            for insight in insights:
                details += f"  • {insight}\n"
        
        return details
    
    @staticmethod
    def _generate_prediction_details(results: Dict[str, Any]) -> str:
        """Generate prediction details"""
        details = """
MODEL PERFORMANCE
{'='*60}

"""
        accuracy = results.get('accuracy', 0)
        model_type = results.get('model_type', 'unknown')
        
        details += f"Model Type: {model_type.title()}\n"
        if model_type == 'classification':
            details += f"Accuracy: {accuracy:.2%}\n\n"
            details += "Interpretation:\n"
            if accuracy > 0.9:
                details += "  ✅ Excellent model performance! The model is highly accurate.\n"
            elif accuracy > 0.7:
                details += "  ✅ Good model performance. The model is reasonably accurate.\n"
            else:
                details += "  ⚠️ Model may need improvement. Consider more data or different features.\n"
        else:
            details += f"R² Score: {accuracy:.4f}\n\n"
            details += "Interpretation:\n"
            if accuracy > 0.9:
                details += "  ✅ Excellent model! Explains most of the variation.\n"
            elif accuracy > 0.7:
                details += "  ✅ Good model. Explains significant variation.\n"
            else:
                details += "  ⚠️ Model may need improvement.\n"
        
        return details
    
    @staticmethod
    def _generate_clustering_details(results: Dict[str, Any]) -> str:
        """Generate clustering details"""
        details = """
CLUSTER ANALYSIS
{'='*60}

"""
        n_clusters = results.get('n_clusters', 0)
        labels = results.get('cluster_labels', [])
        
        details += f"Number of Clusters: {n_clusters}\n"
        details += f"Total Items: {len(labels)}\n\n"
        
        if labels:
            from collections import Counter
            cluster_counts = Counter(labels)
            details += "Cluster Sizes:\n"
            for cluster, count in sorted(cluster_counts.items()):
                percentage = count / len(labels) * 100
                details += f"  • Cluster {cluster}: {count} items ({percentage:.1f}%)\n"
        
        return details
    
    @staticmethod
    def _generate_feature_importance_details(results: Dict[str, Any]) -> str:
        """Generate feature importance details"""
        details = """
FEATURE IMPORTANCE RANKING
{'='*60}

"""
        importance = results.get('feature_importance', {})
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        details += "All Features (ranked by importance):\n\n"
        for i, (feature, score) in enumerate(sorted_features, 1):
            bar_length = int(score * 50)
            bar = '█' * bar_length
            details += f"{i:2d}. {feature:30s} {bar} {score:.4f} ({score*100:.2f}%)\n"
        
        return details


def create_ai_assistant():
    """Create AI assistant instance"""
    return AIPromptSystem()


# Example usage
if __name__ == '__main__':
    assistant = create_ai_assistant()
    
    print(assistant.start_conversation())
    
    # Example conversation
    responses = [
        "1",  # Analyze data
        "C:/Users/MyData/sales.csv",  # Data file
        "Find patterns in sales data"  # Analysis goal
    ]
    
    for response in responses:
        result = assistant.process_user_input(response)
        print(result.get('response', ''))
        print(result.get('next_question', ''))
