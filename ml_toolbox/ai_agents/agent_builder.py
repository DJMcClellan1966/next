"""
Agent Builder - Build custom AI agents with LLMs, RAG, and Knowledge Graphs

Provides a builder pattern for creating specialized agents
"""
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Agent Builder - Build custom AI agents
    
    Supports:
    - Custom knowledge domains
    - Specialized reasoning
    - Domain-specific RAG
    - Custom knowledge graphs
    """
    
    def __init__(self):
        self.agent_config = {
            'name': 'CustomAgent',
            'description': 'Custom AI Agent',
            'capabilities': [],
            'knowledge_domains': [],
            'custom_prompts': {},
            'custom_reasoning': None
        }
    
    def set_name(self, name: str):
        """Set agent name"""
        self.agent_config['name'] = name
        return self
    
    def set_description(self, description: str):
        """Set agent description"""
        self.agent_config['description'] = description
        return self
    
    def add_capability(self, capability: str):
        """Add capability"""
        if capability not in self.agent_config['capabilities']:
            self.agent_config['capabilities'].append(capability)
        return self
    
    def add_knowledge_domain(self, domain: str, knowledge: List[str]):
        """Add knowledge domain"""
        self.agent_config['knowledge_domains'].append({
            'domain': domain,
            'knowledge': knowledge
        })
        return self
    
    def set_custom_prompt(self, task_type: str, prompt_template: str):
        """Set custom prompt template"""
        self.agent_config['custom_prompts'][task_type] = prompt_template
        return self
    
    def set_custom_reasoning(self, reasoning_function: Callable):
        """Set custom reasoning function"""
        self.agent_config['custom_reasoning'] = reasoning_function
        return self
    
    def build(self, toolbox=None):
        """
        Build the agent
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
            
        Returns
        -------
        agent : LLMRAGKGAgent
            Built agent instance
        """
        from .llm_rag_kg_agent import LLMRAGKGAgent
        
        # Create agent
        agent = LLMRAGKGAgent(toolbox=toolbox)
        
        # Add knowledge domains to RAG
        if agent.rag_system:
            for domain_info in self.agent_config['knowledge_domains']:
                domain = domain_info['domain']
                knowledge = domain_info['knowledge']
                for i, text in enumerate(knowledge):
                    doc_id = f"{domain}_{i}"
                    agent.add_knowledge(text, doc_id, add_to_kg=True, add_to_rag=True)
        
        # Add custom prompts
        if agent.llm_components.get('prompt_engineer'):
            for task_type, template in self.agent_config['custom_prompts'].items():
                from ..llm_engineering import PromptTemplate
                agent.llm_components['prompt_engineer'].templates[task_type] = PromptTemplate(template)
        
        logger.info(f"[AgentBuilder] Built agent: {self.agent_config['name']}")
        
        return agent
    
    def build_ml_agent(self, toolbox=None):
        """
        Build specialized ML agent
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
            
        Returns
        -------
        agent : LLMRAGKGAgent
            ML-specialized agent
        """
        # ML-specific knowledge
        ml_knowledge = [
            "Random Forest is an ensemble method that combines multiple decision trees",
            "Support Vector Machines work well for high-dimensional data",
            "Gradient Boosting often achieves the best performance for structured data",
            "Cross-validation is essential for reliable model evaluation",
            "Feature engineering can significantly improve model performance",
            "Hyperparameter tuning is important for optimizing model performance"
        ]
        
        return (self
                .set_name("MLSpecialistAgent")
                .set_description("Specialized agent for machine learning tasks")
                .add_capability("classification")
                .add_capability("regression")
                .add_capability("feature_engineering")
                .add_capability("model_selection")
                .add_knowledge_domain("machine_learning", ml_knowledge)
                .build(toolbox=toolbox))
    
    def build_data_agent(self, toolbox=None):
        """
        Build specialized data analysis agent
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
            
        Returns
        -------
        agent : LLMRAGKGAgent
            Data-specialized agent
        """
        data_knowledge = [
            "Data cleaning is the first step in any ML pipeline",
            "Missing values can be handled by imputation or removal",
            "Outliers should be identified and handled appropriately",
            "Feature scaling is important for distance-based algorithms",
            "Data visualization helps understand patterns and relationships"
        ]
        
        return (self
                .set_name("DataAnalysisAgent")
                .set_description("Specialized agent for data analysis")
                .add_capability("data_cleaning")
                .add_capability("data_visualization")
                .add_capability("statistical_analysis")
                .add_knowledge_domain("data_analysis", data_knowledge)
                .build(toolbox=toolbox))
    
    def build_deployment_agent(self, toolbox=None):
        """
        Build specialized deployment agent
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
            
        Returns
        -------
        agent : LLMRAGKGAgent
            Deployment-specialized agent
        """
        deployment_knowledge = [
            "Model deployment requires careful versioning and monitoring",
            "A/B testing helps compare model versions",
            "Canary deployment allows gradual rollout",
            "Model monitoring detects drift and performance degradation",
            "API design is crucial for production ML systems"
        ]
        
        return (self
                .set_name("DeploymentAgent")
                .set_description("Specialized agent for model deployment")
                .add_capability("model_deployment")
                .add_capability("monitoring")
                .add_capability("a_b_testing")
                .add_knowledge_domain("deployment", deployment_knowledge)
                .build(toolbox=toolbox))
