"""
Code Quality Tools
Linting, formatting, type checking, documentation, and coverage

Features:
- Code linting (pylint, flake8)
- Code formatting (black)
- Type checking (mypy)
- Documentation generation (Sphinx)
- Code coverage reporting
- Performance profiling
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import subprocess
import warnings
import os

sys.path.insert(0, str(Path(__file__).parent))


class CodeLinter:
    """
    Code Linter
    
    Lint code using pylint and flake8
    """
    
    def __init__(self, linter: str = 'pylint'):
        """
        Args:
            linter: Linter to use ('pylint' or 'flake8')
        """
        self.linter = linter
    
    def lint_file(self, file_path: str, options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Lint a single file
        
        Args:
            file_path: Path to file to lint
            options: Additional linter options
            
        Returns:
            Linting results
        """
        if not os.path.exists(file_path):
            return {'error': f'File not found: {file_path}'}
        
        options = options or []
        
        try:
            if self.linter == 'pylint':
                cmd = ['pylint', file_path] + options
            elif self.linter == 'flake8':
                cmd = ['flake8', file_path] + options
            else:
                return {'error': f'Unknown linter: {self.linter}'}
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'file': file_path,
                'linter': self.linter,
                'exit_code': result.returncode,
                'output': result.stdout,
                'errors': result.stderr,
                'passed': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {'error': 'Linting timed out', 'file': file_path}
        except FileNotFoundError:
            return {'error': f'{self.linter} not installed. Install with: pip install {self.linter}'}
        except Exception as e:
            return {'error': str(e), 'file': file_path}
    
    def lint_directory(self, directory: str, pattern: str = '*.py') -> Dict[str, Any]:
        """
        Lint all files in a directory
        
        Args:
            directory: Directory to lint
            pattern: File pattern to match
            
        Returns:
            Linting results for all files
        """
        from glob import glob
        
        files = glob(os.path.join(directory, '**', pattern), recursive=True)
        
        results = {
            'directory': directory,
            'files_linted': len(files),
            'files_passed': 0,
            'files_failed': 0,
            'file_results': []
        }
        
        for file_path in files:
            result = self.lint_file(file_path)
            results['file_results'].append(result)
            
            if result.get('passed', False):
                results['files_passed'] += 1
            else:
                results['files_failed'] += 1
        
        return results


class CodeFormatter:
    """
    Code Formatter
    
    Format code using black
    """
    
    def __init__(self, formatter: str = 'black'):
        """
        Args:
            formatter: Formatter to use ('black' or 'autopep8')
        """
        self.formatter = formatter
    
    def format_file(self, file_path: str, check_only: bool = False) -> Dict[str, Any]:
        """
        Format a single file
        
        Args:
            file_path: Path to file to format
            check_only: Only check if formatting is needed (don't modify)
            
        Returns:
            Formatting results
        """
        if not os.path.exists(file_path):
            return {'error': f'File not found: {file_path}'}
        
        try:
            if self.formatter == 'black':
                cmd = ['black', '--check', file_path] if check_only else ['black', file_path]
            elif self.formatter == 'autopep8':
                cmd = ['autopep8', '--check', file_path] if check_only else ['autopep8', '--in-place', file_path]
            else:
                return {'error': f'Unknown formatter: {self.formatter}'}
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'file': file_path,
                'formatter': self.formatter,
                'exit_code': result.returncode,
                'output': result.stdout,
                'errors': result.stderr,
                'formatted': result.returncode == 0 if check_only else True,
                'needs_formatting': result.returncode != 0 if check_only else False
            }
        except subprocess.TimeoutExpired:
            return {'error': 'Formatting timed out', 'file': file_path}
        except FileNotFoundError:
            return {'error': f'{self.formatter} not installed. Install with: pip install {self.formatter}'}
        except Exception as e:
            return {'error': str(e), 'file': file_path}
    
    def format_directory(self, directory: str, check_only: bool = False) -> Dict[str, Any]:
        """Format all files in a directory"""
        from glob import glob
        
        files = glob(os.path.join(directory, '**', '*.py'), recursive=True)
        
        results = {
            'directory': directory,
            'files_formatted': len(files),
            'files_needing_format': 0,
            'file_results': []
        }
        
        for file_path in files:
            result = self.format_file(file_path, check_only)
            results['file_results'].append(result)
            
            if result.get('needs_formatting', False):
                results['files_needing_format'] += 1
        
        return results


class TypeChecker:
    """
    Type Checker
    
    Check type hints using mypy
    """
    
    def __init__(self, checker: str = 'mypy'):
        """
        Args:
            checker: Type checker to use ('mypy' or 'pyright')
        """
        self.checker = checker
    
    def check_file(self, file_path: str, options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Type check a single file
        
        Args:
            file_path: Path to file to check
            options: Additional checker options
            
        Returns:
            Type checking results
        """
        if not os.path.exists(file_path):
            return {'error': f'File not found: {file_path}'}
        
        options = options or []
        
        try:
            if self.checker == 'mypy':
                cmd = ['mypy', file_path] + options
            elif self.checker == 'pyright':
                cmd = ['pyright', file_path] + options
            else:
                return {'error': f'Unknown type checker: {self.checker}'}
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'file': file_path,
                'checker': self.checker,
                'exit_code': result.returncode,
                'output': result.stdout,
                'errors': result.stderr,
                'passed': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {'error': 'Type checking timed out', 'file': file_path}
        except FileNotFoundError:
            return {'error': f'{self.checker} not installed. Install with: pip install {self.checker}'}
        except Exception as e:
            return {'error': str(e), 'file': file_path}


class DocumentationGenerator:
    """
    Documentation Generator
    
    Generate documentation using Sphinx
    """
    
    def __init__(self, generator: str = 'sphinx'):
        """
        Args:
            generator: Documentation generator ('sphinx' or 'pydoc')
        """
        self.generator = generator
    
    def generate_docs(self, source_dir: str, output_dir: str = 'docs') -> Dict[str, Any]:
        """
        Generate documentation
        
        Args:
            source_dir: Source directory
            output_dir: Output directory for docs
            
        Returns:
            Documentation generation results
        """
        try:
            if self.generator == 'sphinx':
                # Initialize Sphinx if needed
                if not os.path.exists(os.path.join(output_dir, 'conf.py')):
                    subprocess.run(['sphinx-quickstart', output_dir], check=False)
                
                # Build documentation
                result = subprocess.run(
                    ['sphinx-build', '-b', 'html', source_dir, output_dir],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            else:
                return {'error': f'Unknown generator: {self.generator}'}
            
            return {
                'generator': self.generator,
                'source_dir': source_dir,
                'output_dir': output_dir,
                'exit_code': result.returncode,
                'output': result.stdout,
                'errors': result.stderr,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {'error': 'Documentation generation timed out'}
        except FileNotFoundError:
            return {'error': f'{self.generator} not installed. Install with: pip install {self.generator}'}
        except Exception as e:
            return {'error': str(e)}


class CoverageReporter:
    """
    Coverage Reporter
    
    Generate code coverage reports
    """
    
    def __init__(self, tool: str = 'coverage'):
        """
        Args:
            tool: Coverage tool ('coverage' or 'pytest-cov')
        """
        self.tool = tool
    
    def run_coverage(self, test_command: List[str], source: str = '.') -> Dict[str, Any]:
        """
        Run coverage analysis
        
        Args:
            test_command: Command to run tests (e.g., ['pytest', 'tests/'])
            source: Source directory to analyze
            
        Returns:
            Coverage results
        """
        try:
            if self.tool == 'coverage':
                # Run coverage
                result = subprocess.run(
                    ['coverage', 'run', '--source', source] + test_command,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    # Generate report
                    report_result = subprocess.run(
                        ['coverage', 'report'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    # Generate HTML report
                    html_result = subprocess.run(
                        ['coverage', 'html'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    return {
                        'tool': self.tool,
                        'exit_code': result.returncode,
                        'coverage_output': report_result.stdout,
                        'html_report': 'htmlcov/index.html' if html_result.returncode == 0 else None,
                        'success': result.returncode == 0
                    }
                else:
                    return {
                        'tool': self.tool,
                        'exit_code': result.returncode,
                        'error': result.stderr,
                        'success': False
                    }
            else:
                return {'error': f'Unknown coverage tool: {self.tool}'}
        except subprocess.TimeoutExpired:
            return {'error': 'Coverage analysis timed out'}
        except FileNotFoundError:
            return {'error': f'{self.tool} not installed. Install with: pip install {self.tool}'}
        except Exception as e:
            return {'error': str(e)}


class PerformanceProfiler:
    """
    Performance Profiler
    
    Profile code performance using cProfile
    """
    
    def __init__(self):
        """Initialize profiler"""
        pass
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a function
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profiling results
        """
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        return {
            'function': func.__name__,
            'result': result,
            'profile_stats': s.getvalue(),
            'total_calls': ps.total_calls,
            'total_time': ps.total_tt
        }
    
    def profile_file(self, file_path: str, function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Profile a file or specific function
        
        Args:
            file_path: Path to file
            function_name: Specific function to profile (None = entire file)
            
        Returns:
            Profiling results
        """
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        
        try:
            if function_name:
                # Profile specific function
                import importlib.util
                spec = importlib.util.spec_from_file_location("module", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                func = getattr(module, function_name)
                
                profiler.enable()
                result = func()
                profiler.disable()
            else:
                # Profile entire file
                profiler.enable()
                exec(open(file_path).read())
                profiler.disable()
                result = None
        except Exception as e:
            return {'error': str(e), 'file': file_path}
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        return {
            'file': file_path,
            'function': function_name,
            'result': result,
            'profile_stats': s.getvalue(),
            'total_calls': ps.total_calls,
            'total_time': ps.total_tt
        }


class CodeQualitySuite:
    """
    Code Quality Suite
    
    Comprehensive code quality checking
    """
    
    def __init__(self):
        """Initialize code quality suite"""
        self.linter = CodeLinter()
        self.formatter = CodeFormatter()
        self.type_checker = TypeChecker()
        self.coverage = CoverageReporter()
        self.profiler = PerformanceProfiler()
    
    def check_all(self, file_path: str) -> Dict[str, Any]:
        """
        Run all quality checks on a file
        
        Args:
            file_path: Path to file to check
            
        Returns:
            Comprehensive quality report
        """
        results = {
            'file': file_path,
            'linting': self.linter.lint_file(file_path),
            'formatting': self.formatter.format_file(file_path, check_only=True),
            'type_checking': self.type_checker.check_file(file_path),
            'quality_score': 0.0
        }
        
        # Calculate quality score
        score = 0.0
        if results['linting'].get('passed', False):
            score += 25.0
        if not results['formatting'].get('needs_formatting', True):
            score += 25.0
        if results['type_checking'].get('passed', False):
            score += 25.0
        # Coverage would add remaining 25%
        
        results['quality_score'] = score
        
        return results
