"""
AI Prompt Interface
Interactive command-line interface for non-technical users
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import os

sys.path.insert(0, str(Path(__file__).parent))

from ai_prompt_system import AIPromptSystem, GuidedWorkflow, ReportGenerator


class InteractiveAIAssistant:
    """
    Interactive AI Assistant
    
    Command-line interface for non-technical users
    """
    
    def __init__(self):
        """Initialize interactive assistant"""
        self.prompt_system = AIPromptSystem()
        self.workflow = GuidedWorkflow(self.prompt_system)
        self.running = True
    
    def run(self):
        """Run interactive assistant"""
        print("\n" + "="*80)
        print("🤖 ML Toolbox AI Assistant - Interactive Mode")
        print("="*80)
        print("\nType 'help' for commands, 'quit' to exit\n")
        
        # Start conversation
        welcome = self.prompt_system.start_conversation()
        print(welcome)
        
        while self.running:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using ML Toolbox AI Assistant!")
                    self.running = False
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'summary':
                    self._show_summary()
                    continue
                
                # Process input
                result = self.prompt_system.process_user_input(user_input)
                
                # Display response
                if result.get('response'):
                    print(f"\n🤖 Assistant: {result['response']}")
                
                if result.get('next_question'):
                    print(f"\n❓ {result['next_question']}")
                
                if result.get('report'):
                    print("\n" + "="*80)
                    print(result['report'])
                    print("="*80)
                
                # Handle data loading
                if 'file' in user_input.lower() or user_input.endswith('.csv') or user_input.endswith('.xlsx'):
                    if os.path.exists(user_input):
                        load_result = self.prompt_system.load_data(user_input)
                        print(f"\n🤖 Assistant: {load_result.get('message', '')}")
                        if load_result.get('success'):
                            info = load_result.get('data_info', {})
                            print(f"   📊 Found {info.get('rows', 0)} rows and {info.get('columns', 0)} columns")
                
                # Check if ready to execute
                if result.get('status') == 'ready_to_execute':
                    execute = input("\n✅ Ready to run analysis? (yes/no): ").strip().lower()
                    if execute in ['yes', 'y', 'ok']:
                        exec_result = self.prompt_system._execute_task({'type': 'execute'})
                        if exec_result.get('report'):
                            print("\n" + "="*80)
                            print(exec_result['report'])
                            print("="*80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'help' for assistance.")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
📚 HELP - Available Commands:

Commands:
  help          - Show this help message
  summary       - Show conversation summary
  quit/exit/q   - Exit the assistant

How to Use:
  1. Answer the questions I ask
  2. Provide your data file path when asked
  3. Describe what you want to do
  4. I'll guide you through the process

Example:
  👤 You: 1
  🤖 Assistant: Great! I'll help you with data analysis...
  👤 You: C:/Users/MyData/sales.csv
  🤖 Assistant: ✅ Data loaded successfully!
  ...
"""
        print(help_text)
    
    def _show_summary(self):
        """Show conversation summary"""
        summary = self.prompt_system.get_conversation_summary()
        
        print("\n" + "="*80)
        print("📋 CONVERSATION SUMMARY")
        print("="*80)
        print(f"Total Messages: {summary['total_messages']}")
        print(f"Current Task: {summary['current_task'] or 'None'}")
        print(f"Data Loaded: {'Yes' if summary['data_loaded'] else 'No'}")
        
        if summary['data_loaded']:
            info = summary['data_info']
            print(f"  Rows: {info.get('rows', 'N/A')}")
            print(f"  Columns: {info.get('columns', 'N/A')}")
        
        print("="*80)


def main():
    """Main entry point"""
    assistant = InteractiveAIAssistant()
    assistant.run()


if __name__ == '__main__':
    main()
