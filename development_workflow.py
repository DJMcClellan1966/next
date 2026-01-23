"""
Development Workflow Tools
Pre-commit hooks, CI/CD, code review automation

Features:
- Pre-commit hooks
- CI/CD pipeline configuration
- Code review automation
- Quality gates
- Release management
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import os
import json
import subprocess
import warnings

sys.path.insert(0, str(Path(__file__).parent))


class PreCommitHooks:
    """
    Pre-commit Hooks
    
    Set up and manage pre-commit hooks
    """
    
    def __init__(self, hooks_dir: str = '.git/hooks'):
        """
        Args:
            hooks_dir: Directory for git hooks
        """
        self.hooks_dir = Path(hooks_dir)
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
    
    def create_pre_commit_hook(self, checks: List[str]) -> Dict[str, Any]:
        """
        Create pre-commit hook
        
        Args:
            checks: List of checks to run ('lint', 'format', 'type', 'test')
            
        Returns:
            Hook creation result
        """
        hook_content = "#!/bin/sh\n"
        hook_content += "# Pre-commit hook for ML Toolbox\n\n"
        
        if 'lint' in checks:
            hook_content += "echo 'Running linter...'\n"
            hook_content += "pylint --errors-only ml_toolbox/ || exit 1\n\n"
        
        if 'format' in checks:
            hook_content += "echo 'Checking code formatting...'\n"
            hook_content += "black --check ml_toolbox/ || exit 1\n\n"
        
        if 'type' in checks:
            hook_content += "echo 'Running type checker...'\n"
            hook_content += "mypy ml_toolbox/ || exit 1\n\n"
        
        if 'test' in checks:
            hook_content += "echo 'Running tests...'\n"
            hook_content += "pytest tests/ -x || exit 1\n\n"
        
        hook_content += "echo 'All checks passed!'\n"
        
        hook_path = self.hooks_dir / 'pre-commit'
        
        try:
            with open(hook_path, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            os.chmod(hook_path, 0o755)
            
            return {
                'success': True,
                'hook_path': str(hook_path),
                'checks': checks,
                'message': 'Pre-commit hook created successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_hook_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create hook configuration file
        
        Args:
            config: Hook configuration
            
        Returns:
            Config creation result
        """
        config_path = Path('.pre-commit-config.yaml')
        
        yaml_content = "repos:\n"
        yaml_content += "  - repo: local\n"
        yaml_content += "    hooks:\n"
        
        if config.get('lint', False):
            yaml_content += "      - id: pylint\n"
            yaml_content += "        name: pylint\n"
            yaml_content += "        entry: pylint\n"
            yaml_content += "        language: system\n"
            yaml_content += "        args: [--errors-only, ml_toolbox/]\n"
        
        if config.get('format', False):
            yaml_content += "      - id: black\n"
            yaml_content += "        name: black\n"
            yaml_content += "        entry: black\n"
            yaml_content += "        language: system\n"
            yaml_content += "        args: [--check, ml_toolbox/]\n"
        
        try:
            with open(config_path, 'w') as f:
                f.write(yaml_content)
            
            return {
                'success': True,
                'config_path': str(config_path),
                'message': 'Hook configuration created'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class CICDPipeline:
    """
    CI/CD Pipeline
    
    Configure CI/CD pipeline
    """
    
    def create_github_actions(self, workflows: List[str]) -> Dict[str, Any]:
        """
        Create GitHub Actions workflow
        
        Args:
            workflows: List of workflows ('test', 'lint', 'deploy')
            
        Returns:
            Workflow creation result
        """
        workflows_dir = Path('.github/workflows')
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_content = "name: ML Toolbox CI/CD\n\n"
        workflow_content += "on:\n"
        workflow_content += "  push:\n"
        workflow_content += "    branches: [ main ]\n"
        workflow_content += "  pull_request:\n"
        workflow_content += "    branches: [ main ]\n\n"
        workflow_content += "jobs:\n"
        
        if 'test' in workflows:
            workflow_content += "  test:\n"
            workflow_content += "    runs-on: ubuntu-latest\n"
            workflow_content += "    steps:\n"
            workflow_content += "      - uses: actions/checkout@v2\n"
            workflow_content += "      - name: Set up Python\n"
            workflow_content += "        uses: actions/setup-python@v2\n"
            workflow_content += "        with:\n"
            workflow_content += "          python-version: '3.11'\n"
            workflow_content += "      - name: Install dependencies\n"
            workflow_content += "        run: pip install -r requirements.txt\n"
            workflow_content += "      - name: Run tests\n"
            workflow_content += "        run: pytest tests/ -v\n\n"
        
        if 'lint' in workflows:
            workflow_content += "  lint:\n"
            workflow_content += "    runs-on: ubuntu-latest\n"
            workflow_content += "    steps:\n"
            workflow_content += "      - uses: actions/checkout@v2\n"
            workflow_content += "      - name: Set up Python\n"
            workflow_content += "        uses: actions/setup-python@v2\n"
            workflow_content += "        with:\n"
            workflow_content += "          python-version: '3.11'\n"
            workflow_content += "      - name: Install linting tools\n"
            workflow_content += "        run: pip install pylint black mypy\n"
            workflow_content += "      - name: Run linter\n"
            workflow_content += "        run: pylint ml_toolbox/\n"
            workflow_content += "      - name: Check formatting\n"
            workflow_content += "        run: black --check ml_toolbox/\n\n"
        
        workflow_path = workflows_dir / 'ci.yml'
        
        try:
            with open(workflow_path, 'w') as f:
                f.write(workflow_content)
            
            return {
                'success': True,
                'workflow_path': str(workflow_path),
                'workflows': workflows,
                'message': 'GitHub Actions workflow created'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_quality_gates(self, gates: Dict[str, float]) -> Dict[str, Any]:
        """
        Create quality gates
        
        Args:
            gates: Dictionary of gate_name -> threshold (e.g., {'coverage': 0.8, 'quality_score': 0.7})
            
        Returns:
            Quality gates configuration
        """
        gates_config = {
            'gates': gates,
            'enforcement': 'strict'
        }
        
        config_path = Path('.quality_gates.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(gates_config, f, indent=2)
            
            return {
                'success': True,
                'config_path': str(config_path),
                'gates': gates,
                'message': 'Quality gates configured'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class CodeReviewAutomation:
    """
    Code Review Automation
    
    Automated code review checks
    """
    
    def __init__(self):
        """Initialize code review automation"""
        pass
    
    def review_code(self, file_path: str) -> Dict[str, Any]:
        """
        Automated code review
        
        Args:
            file_path: Path to file to review
            
        Returns:
            Review results
        """
        from code_quality_tools import CodeQualitySuite
        
        suite = CodeQualitySuite()
        quality_report = suite.check_all(file_path)
        
        review = {
            'file': file_path,
            'quality_score': quality_report.get('quality_score', 0),
            'issues': [],
            'recommendations': []
        }
        
        # Check linting
        if not quality_report['linting'].get('passed', False):
            review['issues'].append({
                'type': 'linting',
                'severity': 'high',
                'message': 'Linting errors found'
            })
        
        # Check formatting
        if quality_report['formatting'].get('needs_formatting', False):
            review['issues'].append({
                'type': 'formatting',
                'severity': 'medium',
                'message': 'Code needs formatting'
            })
        
        # Check type checking
        if not quality_report['type_checking'].get('passed', False):
            review['issues'].append({
                'type': 'type_checking',
                'severity': 'medium',
                'message': 'Type checking errors found'
            })
        
        # Recommendations
        if review['quality_score'] < 50:
            review['recommendations'].append('Improve code quality - add type hints, fix linting errors')
        
        if review['quality_score'] < 75:
            review['recommendations'].append('Code quality is acceptable but could be improved')
        
        return review


class ReleaseManager:
    """
    Release Manager
    
    Manage releases and versioning
    """
    
    def __init__(self, version_file: str = 'VERSION'):
        """
        Args:
            version_file: Path to version file
        """
        self.version_file = Path(version_file)
    
    def get_version(self) -> str:
        """Get current version"""
        if self.version_file.exists():
            return self.version_file.read_text().strip()
        return '0.0.0'
    
    def bump_version(self, part: str = 'patch') -> Dict[str, Any]:
        """
        Bump version
        
        Args:
            part: Version part to bump ('major', 'minor', 'patch')
            
        Returns:
            Version bump result
        """
        current = self.get_version()
        parts = current.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        if part == 'major':
            major += 1
            minor = 0
            patch = 0
        elif part == 'minor':
            minor += 1
            patch = 0
        else:
            patch += 1
        
        new_version = f'{major}.{minor}.{patch}'
        
        try:
            self.version_file.write_text(new_version)
            
            return {
                'success': True,
                'old_version': current,
                'new_version': new_version,
                'bumped': part
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_release_notes(self, version: str, changes: List[str]) -> Dict[str, Any]:
        """
        Create release notes
        
        Args:
            version: Version number
            changes: List of changes
            
        Returns:
            Release notes creation result
        """
        notes = f"# Release {version}\n\n"
        notes += f"## Changes\n\n"
        
        for change in changes:
            notes += f"- {change}\n"
        
        notes_path = Path(f'RELEASE_NOTES_{version}.md')
        
        try:
            notes_path.write_text(notes)
            
            return {
                'success': True,
                'notes_path': str(notes_path),
                'version': version
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
