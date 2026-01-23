# AI Prompt System Guide

## 🤖 **Overview**

The AI Prompt System is a natural language interface that makes the ML Toolbox accessible to **non-technical users**. No coding required - just answer questions in plain English!

---

## 🚀 **Quick Start**

### **Option 1: Interactive Command-Line Interface**

```bash
python ai_prompt_interface.py
```

This starts an interactive conversation where you can:
- Answer questions in natural language
- Upload your data files
- Get guided through the analysis process
- Receive human-readable reports

### **Option 2: Programmatic Use**

```python
from ai_prompt_system import AIPromptSystem

# Create assistant
assistant = AIPromptSystem()

# Start conversation
print(assistant.start_conversation())

# Process user input
result = assistant.process_user_input("1")  # Analyze data
print(result['response'])
print(result['next_question'])

# Load data
load_result = assistant.load_data("path/to/your/data.csv")
print(load_result['message'])

# Run analysis
results = assistant._run_analysis()
report = assistant._generate_human_readable_report(results)
print(report)
```

---

## 📋 **Supported Tasks**

### **1. Data Analysis**
Find patterns, trends, and insights in your data.

**Example Conversation:**
```
👤 You: 1
🤖 Assistant: Great! I'll help you with data analysis...
👤 You: C:/Users/MyData/sales.csv
🤖 Assistant: ✅ Data loaded successfully! Found 1000 rows and 5 columns.
👤 You: Find patterns in sales data
🤖 Assistant: ✅ Analysis complete! Here are your results...
```

### **2. Prediction**
Build models to predict future values or outcomes.

**Example Conversation:**
```
👤 You: 2
🤖 Assistant: Great! I'll help you with predicting future values...
👤 You: C:/Users/MyData/customers.csv
🤖 Assistant: ✅ Data loaded successfully!
👤 You: Predict customer churn
🤖 Assistant: What column contains the churn information?
👤 You: churn
🤖 Assistant: ✅ Model trained! Accuracy: 92.5%
```

### **3. Clustering**
Group similar items together.

**Example Conversation:**
```
👤 You: 3
🤖 Assistant: Great! I'll help you group similar items...
👤 You: C:/Users/MyData/products.csv
🤖 Assistant: ✅ Data loaded successfully!
👤 You: 3 groups
🤖 Assistant: ✅ Clustering complete! Found 3 distinct groups...
```

### **4. Feature Importance**
Understand which factors are most important.

**Example Conversation:**
```
👤 You: 4
🤖 Assistant: Great! I'll help you understand important factors...
👤 You: C:/Users/MyData/data.csv
🤖 Assistant: ✅ Data loaded successfully!
👤 You: sales
🤖 Assistant: ✅ Analysis complete! Top factors: price, region, season...
```

---

## 💬 **Natural Language Understanding**

The system understands natural language, so you can:

✅ **Say it naturally:**
- "I want to predict sales"
- "Find patterns in my data"
- "Group similar customers together"
- "What factors are important?"

✅ **Answer questions:**
- "Yes, I have a CSV file"
- "The file is at C:/Users/MyData/data.csv"
- "I want to predict customer churn"
- "Use all columns"

✅ **Ask for help:**
- "What can you do?"
- "How does this work?"
- "What do I need to provide?"

---

## 📊 **Data Requirements**

### **Supported File Formats:**
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

### **Data Structure:**
Your data should be in a table format with:
- **Rows:** Individual records/observations
- **Columns:** Features/variables

**Example:**
```
sales, date, region, product
1000, 2024-01-01, North, Widget A
1500, 2024-01-02, South, Widget B
...
```

---

## 🎯 **Guided Workflows**

The system uses **question-based prompts** to guide you:

### **Step 1: Task Selection**
```
📋 What would you like to do today?

1. Analyze data and find patterns
2. Predict future values or outcomes
3. Group similar items together
4. Understand what factors are important
5. Something else (describe it)
```

### **Step 2: Data Information**
```
📊 Tell me about your data:

1. Do you have a data file ready? (CSV, Excel, etc.)
2. What does your data contain?
3. What are you trying to predict or analyze?
```

### **Step 3: Parameters**
```
🎯 What do you want to predict? (name the column)
💡 Example: "sales" or "customer_churn" or "price"
```

---

## 📈 **Report Generation**

### **Executive Summary**
Every analysis includes:
- **Task Type:** What was analyzed
- **Key Findings:** Main insights
- **Date:** When analysis was run

### **Detailed Reports**
Reports include:
- **Data Summary:** Rows, columns, data types
- **Analysis Results:** Metrics, accuracy, scores
- **Insights:** Key observations
- **Interpretation:** What results mean in plain English

### **Example Report:**
```
================================================================================
📊 ANALYSIS REPORT
================================================================================

Task: Prediction
Generated: 2024-01-20 15:30:00

🎯 PREDICTION MODEL RESULTS:

Model Type: Classification
Target Column: customer_churn
Accuracy: 92.50%

💡 INTERPRETATION:
- Higher accuracy means better predictions
- The model learned patterns from your data
- You can use this model to predict new values
```

---

## 🛠️ **Advanced Features**

### **Guided Workflow**
For step-by-step guidance:

```python
from ai_prompt_system import GuidedWorkflow, AIPromptSystem

system = AIPromptSystem()
workflow = GuidedWorkflow(system)

# Create workflow for prediction
steps = workflow.create_workflow('prediction')

# Process each step
for step in steps:
    print(step['question'])
    response = input("Your answer: ")
    result = workflow.process_step_response(step['step'], response)
```

### **Custom Report Generation**
Generate custom reports:

```python
from ai_prompt_system import ReportGenerator

results = {
    'task': 'prediction',
    'accuracy': 0.95,
    'model_type': 'classification'
}

# Executive summary
summary = ReportGenerator.generate_executive_summary(results)

# Detailed report
report = ReportGenerator.generate_detailed_report(results)
```

---

## 💡 **Best Practices**

### **For Users:**
1. ✅ **Be specific:** "Predict sales next month" vs "predict"
2. ✅ **Provide context:** Describe your data and goals
3. ✅ **Answer questions:** The system will ask what it needs
4. ✅ **Ask for help:** Type "help" anytime

### **For Developers:**
1. ✅ **Use clear language:** Avoid technical jargon
2. ✅ **Provide examples:** Show users what to expect
3. ✅ **Handle errors gracefully:** Give helpful error messages
4. ✅ **Track conversation:** Maintain context throughout

---

## 🔧 **Troubleshooting**

### **"Data file not found"**
- Check the file path is correct
- Use full path: `C:/Users/MyData/file.csv`
- Make sure file exists

### **"Target column not found"**
- Check column name spelling
- Column names are case-sensitive
- List columns: The system will show available columns

### **"Analysis failed"**
- Check data format (CSV/Excel)
- Ensure numeric columns for predictions
- Check for missing data

### **"I don't understand"**
- Rephrase your request
- Use simpler language
- Answer the questions asked

---

## 📚 **Examples**

### **Example 1: Sales Prediction**
```
👤 You: 2
🤖 Assistant: Great! I'll help you with predicting future values...
👤 You: C:/Data/sales.csv
🤖 Assistant: ✅ Data loaded! Found 500 rows and 4 columns.
👤 You: Predict sales
🤖 Assistant: What column contains sales? (or describe it)
👤 You: sales_amount
🤖 Assistant: ✅ Model trained! R² Score: 0.95
```

### **Example 2: Customer Segmentation**
```
👤 You: 3
🤖 Assistant: Great! I'll help you group similar items...
👤 You: C:/Data/customers.csv
🤖 Assistant: ✅ Data loaded!
👤 You: 5 groups
🤖 Assistant: ✅ Clustering complete! Found 5 distinct customer segments...
```

---

## 🎓 **Learning More**

### **Available Commands:**
- `help` - Show help message
- `summary` - Show conversation summary
- `quit` - Exit the assistant

### **Integration with ML Toolbox:**
The AI Prompt System integrates with all ML Toolbox features:
- Data preprocessing
- Model training
- Evaluation metrics
- Feature selection
- And more!

---

## 📞 **Support**

For questions or issues:
1. Check this guide
2. Use the `help` command in the interface
3. Review example conversations above

---

**Happy Analyzing! 🚀**
