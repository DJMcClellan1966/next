"""
Data Collection Pipeline

ETL Pattern: Extract → Transform → Load
- Extract: From user inputs and NoSQL databases
- Transform: Clean, validate, structure data
- Load: Output to Feature Pipeline
"""
import numpy as np
from typing import Any, Dict, Optional, List, Union
import logging
from datetime import datetime

from .base import BasePipeline, PipelineStage

logger = logging.getLogger(__name__)


class ExtractStage(PipelineStage):
    """Stage 1: Extract - Extract data from various sources"""
    
    def __init__(self, toolbox=None):
        super().__init__("extract")
        self.toolbox = toolbox
    
    def execute(self, input_data: Union[Dict, List, str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Extract data from user inputs and NoSQL databases
        
        Parameters
        ----------
        input_data : dict, list, str, or array-like
            Input data from user or NoSQL query result
        **kwargs
            Additional parameters:
            - source_type: 'user_input', 'nosql', 'auto'
            - nosql_query: Query for NoSQL database
            - nosql_collection: Collection/table name
            - nosql_client: NoSQL database client
            
        Returns
        -------
        extracted_data : dict
            Extracted data with metadata
        """
        source_type = kwargs.get('source_type', 'auto')
        nosql_query = kwargs.get('nosql_query')
        nosql_collection = kwargs.get('nosql_collection')
        nosql_client = kwargs.get('nosql_client')
        
        # Auto-detect source type
        if source_type == 'auto':
            if isinstance(input_data, dict) and 'query' in input_data:
                source_type = 'nosql'
            elif isinstance(input_data, (list, dict, str)):
                source_type = 'user_input'
            else:
                source_type = 'user_input'
        
        extracted_data = {
            'source_type': source_type,
            'raw_data': input_data,
            'extraction_timestamp': datetime.now().isoformat(),
            'metadata': {}
        }
        
        # Extract from user input
        if source_type == 'user_input':
            extracted_data['data'] = self._extract_from_user_input(input_data)
            extracted_data['metadata']['source'] = 'user_input'
            extracted_data['metadata']['data_type'] = type(input_data).__name__
        
        # Extract from NoSQL database
        elif source_type == 'nosql':
            if nosql_client and nosql_collection:
                extracted_data['data'] = self._extract_from_nosql(
                    nosql_client, nosql_collection, nosql_query, input_data
                )
                extracted_data['metadata']['source'] = 'nosql'
                extracted_data['metadata']['collection'] = nosql_collection
            else:
                # Fallback: treat input_data as NoSQL result
                extracted_data['data'] = self._extract_from_nosql_result(input_data)
                extracted_data['metadata']['source'] = 'nosql_result'
        
        # Record extraction metadata
        if isinstance(extracted_data['data'], np.ndarray):
            self.metadata['extracted_shape'] = extracted_data['data'].shape
            self.metadata['extracted_dtype'] = str(extracted_data['data'].dtype)
        elif isinstance(extracted_data['data'], (list, dict)):
            self.metadata['extracted_length'] = len(extracted_data['data'])
        
        logger.info(f"[ExtractStage] Extracted data from {source_type}: {self.metadata}")
        
        return extracted_data
    
    def _extract_from_user_input(self, input_data: Any) -> np.ndarray:
        """Extract data from user input"""
        if isinstance(input_data, np.ndarray):
            return input_data
        elif isinstance(input_data, list):
            # Convert list to numpy array
            return np.asarray(input_data)
        elif isinstance(input_data, dict):
            # Extract values from dictionary
            if 'data' in input_data:
                data = input_data['data']
            elif 'features' in input_data:
                data = input_data['features']
            else:
                # Convert dict values to array
                data = list(input_data.values())
            
            return np.asarray(data)
        elif isinstance(input_data, str):
            # Try to parse as JSON or CSV-like
            try:
                import json
                parsed = json.loads(input_data)
                return self._extract_from_user_input(parsed)
            except:
                # Treat as single value
                return np.array([[float(input_data)]])
        else:
            # Convert to array
            return np.asarray([input_data])
    
    def _extract_from_nosql(self, client: Any, collection: str, query: Optional[Dict], 
                            input_data: Any) -> np.ndarray:
        """Extract data from NoSQL database"""
        try:
            # Try common NoSQL patterns
            if hasattr(client, 'find'):
                # MongoDB-like
                if query:
                    results = list(client[collection].find(query))
                else:
                    results = list(client[collection].find())
            elif hasattr(client, 'query'):
                # DynamoDB-like or custom query interface
                if query:
                    results = client.query(collection, query)
                else:
                    results = client.query(collection, {})
            elif hasattr(client, 'get'):
                # Simple key-value store
                if query:
                    results = [client.get(collection, query)]
                else:
                    results = list(client.get_all(collection).values()) if hasattr(client, 'get_all') else []
            else:
                # Fallback: use input_data as result
                results = input_data if isinstance(input_data, list) else [input_data]
            
            # Convert NoSQL results to array
            return self._convert_nosql_results_to_array(results)
            
        except Exception as e:
            logger.warning(f"[ExtractStage] NoSQL extraction failed: {e}. Using input_data as fallback.")
            return self._extract_from_nosql_result(input_data)
    
    def _extract_from_nosql_result(self, nosql_result: Any) -> np.ndarray:
        """Extract data from NoSQL query result"""
        if isinstance(nosql_result, list):
            return self._convert_nosql_results_to_array(nosql_result)
        elif isinstance(nosql_result, dict):
            # Single document
            return self._convert_nosql_document_to_array(nosql_result)
        else:
            return np.asarray([nosql_result])
    
    def _convert_nosql_results_to_array(self, results: List[Dict]) -> np.ndarray:
        """Convert NoSQL results (list of documents) to numpy array"""
        if not results:
            return np.array([])
        
        # Extract numeric fields from documents
        arrays = []
        for doc in results:
            array = self._convert_nosql_document_to_array(doc)
            arrays.append(array.flatten())
        
        if arrays:
            return np.array(arrays)
        else:
            return np.array([])
    
    def _convert_nosql_document_to_array(self, doc: Dict) -> np.ndarray:
        """Convert a single NoSQL document to numpy array"""
        # Extract numeric values, skip _id and metadata fields
        numeric_values = []
        skip_fields = {'_id', 'id', 'timestamp', 'created_at', 'updated_at', 'metadata'}
        
        for key, value in doc.items():
            if key not in skip_fields:
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
                elif isinstance(value, list):
                    # Flatten list of numbers
                    numeric_values.extend([v for v in value if isinstance(v, (int, float))])
        
        if numeric_values:
            return np.array(numeric_values)
        else:
            # Fallback: convert all values
            values = [v for v in doc.values() if isinstance(v, (int, float))]
            return np.array(values) if values else np.array([0])


class TransformStage(PipelineStage):
    """Stage 2: Transform - Clean, validate, and structure data"""
    
    def __init__(self, toolbox=None):
        super().__init__("transform")
        self.toolbox = toolbox
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Transform extracted data
        
        Parameters
        ----------
        input_data : dict
            Extracted data from Extract stage
        **kwargs
            Additional parameters:
            - remove_nulls: Remove null/NaN values (default: True)
            - handle_missing: Strategy for missing values ('drop', 'fill_mean', 'fill_zero')
            - normalize_types: Normalize data types (default: True)
            - validate_schema: Validate data schema (default: False)
            
        Returns
        -------
        transformed_data : dict
            Transformed data ready for loading
        """
        raw_data = input_data.get('data')
        if raw_data is None:
            raise ValueError("No data to transform")
        
        remove_nulls = kwargs.get('remove_nulls', True)
        handle_missing = kwargs.get('handle_missing', 'drop')
        normalize_types = kwargs.get('normalize_types', True)
        
        # Convert to numpy array if not already
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.asarray(raw_data)
        
        # Handle nulls/NaNs
        if remove_nulls:
            raw_data = self._remove_nulls(raw_data)
        
        # Handle missing values
        if handle_missing != 'drop':
            raw_data = self._handle_missing_values(raw_data, handle_missing)
        
        # Normalize types
        if normalize_types:
            raw_data = self._normalize_types(raw_data)
        
        # Validate data
        self._validate_data(raw_data)
        
        transformed_data = {
            'data': raw_data,
            'source_type': input_data.get('source_type'),
            'extraction_timestamp': input_data.get('extraction_timestamp'),
            'transformation_timestamp': datetime.now().isoformat(),
            'metadata': {
                **input_data.get('metadata', {}),
                'transformed_shape': raw_data.shape,
                'transformed_dtype': str(raw_data.dtype),
                'nulls_removed': remove_nulls,
                'missing_handled': handle_missing
            }
        }
        
        self.metadata['transformed_shape'] = raw_data.shape
        self.metadata['transformed_dtype'] = str(raw_data.dtype)
        
        logger.info(f"[TransformStage] Transformed data: {raw_data.shape}")
        
        return transformed_data
    
    def _remove_nulls(self, data: np.ndarray) -> np.ndarray:
        """Remove null/NaN values"""
        if data.size == 0:
            return data
        
        # Check if data is numeric
        if not np.issubdtype(data.dtype, np.number):
            # For non-numeric data, just return as-is
            return data
        
        # Remove rows with any NaN
        if data.ndim > 1:
            mask = ~np.isnan(data).any(axis=1)
            return data[mask]
        else:
            mask = ~np.isnan(data)
            return data[mask]
    
    def _handle_missing_values(self, data: np.ndarray, strategy: str) -> np.ndarray:
        """Handle missing values"""
        if data.size == 0:
            return data
        
        # Check if data is numeric
        if not np.issubdtype(data.dtype, np.number):
            return data
        
        if strategy == 'fill_mean':
            if data.ndim > 1:
                for i in range(data.shape[1]):
                    col = data[:, i]
                    if np.issubdtype(col.dtype, np.number):
                        col[np.isnan(col)] = np.nanmean(col)
            else:
                if np.issubdtype(data.dtype, np.number):
                    data[np.isnan(data)] = np.nanmean(data)
        elif strategy == 'fill_zero':
            if np.issubdtype(data.dtype, np.number):
                data = np.nan_to_num(data, nan=0.0)
        
        return data
    
    def _normalize_types(self, data: np.ndarray) -> np.ndarray:
        """Normalize data types"""
        # Convert to float if mixed types
        if data.dtype == object:
            try:
                return data.astype(float)
            except:
                return data
        
        # Ensure numeric type
        if not np.issubdtype(data.dtype, np.number):
            try:
                return data.astype(float)
            except:
                return data
        
        return data
    
    def _validate_data(self, data: np.ndarray):
        """Validate transformed data"""
        if data.size == 0:
            raise ValueError("Transformed data is empty")
        
        if not np.issubdtype(data.dtype, np.number):
            logger.warning(f"[TransformStage] Data type is not numeric: {data.dtype}")


class LoadStage(PipelineStage):
    """Stage 3: Load - Output to Feature Pipeline"""
    
    def __init__(self, toolbox=None, feature_pipeline=None):
        super().__init__("load")
        self.toolbox = toolbox
        self.feature_pipeline = feature_pipeline
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> np.ndarray:
        """
        Load transformed data to Feature Pipeline
        
        Parameters
        ----------
        input_data : dict
            Transformed data from Transform stage
        **kwargs
            Additional parameters:
            - feature_pipeline: FeaturePipeline instance (if not set in __init__)
            - feature_name: Name for features in feature store
            - validate_before_load: Validate data before loading (default: True)
            
        Returns
        -------
        loaded_data : array-like
            Data ready for Feature Pipeline
        """
        transformed_data = input_data.get('data')
        if transformed_data is None:
            raise ValueError("No transformed data to load")
        
        validate_before_load = kwargs.get('validate_before_load', True)
        
        # Final validation
        if validate_before_load:
            self._validate_for_feature_pipeline(transformed_data)
        
        # Prepare data for feature pipeline
        loaded_data = transformed_data
        
        # Store metadata for feature pipeline
        self.metadata['loaded_shape'] = loaded_data.shape
        self.metadata['loaded_dtype'] = str(loaded_data.dtype)
        self.metadata['source_type'] = input_data.get('source_type')
        self.metadata['extraction_timestamp'] = input_data.get('extraction_timestamp')
        self.metadata['transformation_timestamp'] = input_data.get('transformation_timestamp')
        
        logger.info(f"[LoadStage] Loaded data to Feature Pipeline: {loaded_data.shape}")
        
        return loaded_data
    
    def _validate_for_feature_pipeline(self, data: np.ndarray):
        """Validate data is ready for Feature Pipeline"""
        if data.size == 0:
            raise ValueError("Data is empty, cannot load to Feature Pipeline")
        
        if data.ndim == 0:
            raise ValueError("Data must be at least 1D for Feature Pipeline")
        
        if not np.issubdtype(data.dtype, np.number):
            logger.warning(f"[LoadStage] Data type is not numeric: {data.dtype}")


class DataCollectionPipeline(BasePipeline):
    """
    Data Collection Pipeline (ETL Pattern)
    
    Orchestrates:
    1. Extract - From user inputs and NoSQL databases
    2. Transform - Clean, validate, structure data
    3. Load - Output to Feature Pipeline
    """
    
    def __init__(self, toolbox=None, feature_pipeline=None,
                 enable_monitoring: bool = True, enable_retry: bool = False,
                 enable_debugging: bool = False):
        """
        Initialize data collection pipeline
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        feature_pipeline : FeaturePipeline, optional
            Feature Pipeline instance (for direct integration)
        enable_monitoring : bool, default=True
            Enable pipeline monitoring
        enable_retry : bool, default=False
            Enable retry logic
        enable_debugging : bool, default=False
            Enable debugging
        """
        super().__init__("data_collection_pipeline", toolbox,
                        enable_monitoring=enable_monitoring,
                        enable_retry=enable_retry,
                        enable_debugging=enable_debugging)
        
        # Add ETL stages
        self.add_stage(ExtractStage(toolbox))
        self.add_stage(TransformStage(toolbox))
        self.add_stage(LoadStage(toolbox, feature_pipeline))
        
        self.feature_pipeline = feature_pipeline
    
    def execute(self, input_data: Union[Dict, List, str, np.ndarray],
                source_type: str = 'auto', nosql_client: Optional[Any] = None,
                nosql_collection: Optional[str] = None, nosql_query: Optional[Dict] = None,
                feature_name: str = "collected_features", **kwargs) -> np.ndarray:
        """
        Execute ETL pipeline
        
        Parameters
        ----------
        input_data : dict, list, str, or array-like
            Input data from user or NoSQL query
        source_type : str, default='auto'
            Source type: 'user_input', 'nosql', 'auto'
        nosql_client : Any, optional
            NoSQL database client
        nosql_collection : str, optional
            NoSQL collection/table name
        nosql_query : dict, optional
            NoSQL query
        feature_name : str, default="collected_features"
            Name for features in feature store
        **kwargs
            Additional parameters for stages
            
        Returns
        -------
        loaded_data : array-like
            Data ready for Feature Pipeline
        """
        # Start monitoring if enabled
        if self.monitor:
            metrics = self.monitor.start_pipeline(self.name)
        
        # Execute ETL stages sequentially
        result = input_data
        for stage in self.stages:
            if stage.enabled:
                result = stage.run(result, monitor=self.monitor, retry_handler=self.retry_handler,
                                  debugger=self.debugger, source_type=source_type,
                                  nosql_client=nosql_client, nosql_collection=nosql_collection,
                                  nosql_query=nosql_query, feature_name=feature_name, **kwargs)
                
                # Store stage state
                if isinstance(result, dict):
                    self.state[stage.name] = {
                        'metadata': stage.metadata
                    }
                elif hasattr(result, 'shape'):
                    self.state[stage.name] = {
                        'output_shape': result.shape,
                        'metadata': stage.metadata
                    }
        
        # End monitoring if enabled
        if self.monitor and self.monitor.current_metrics:
            self.monitor.end_pipeline()
        
        # Store final data in state
        if isinstance(result, np.ndarray):
            self.state['final_data'] = result
        else:
            # Extract data from dict if needed
            result = result if isinstance(result, np.ndarray) else result.get('data', result)
            self.state['final_data'] = result
        
        self.state['feature_name'] = feature_name
        
        logger.info(f"[DataCollectionPipeline] ETL pipeline completed. Output shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        return result
    
    def execute_to_feature_pipeline(self, input_data: Union[Dict, List, str, np.ndarray],
                                    feature_name: str = "collected_features", **kwargs) -> np.ndarray:
        """
        Execute ETL and directly feed to Feature Pipeline
        
        Parameters
        ----------
        input_data : dict, list, str, or array-like
            Input data
        feature_name : str, default="collected_features"
            Name for features
        **kwargs
            Additional parameters
            
        Returns
        -------
        features : array-like
            Processed features from Feature Pipeline
        """
        # Execute ETL
        collected_data = self.execute(input_data, feature_name=feature_name, **kwargs)
        
        # Feed to Feature Pipeline if available
        if self.feature_pipeline:
            features = self.feature_pipeline.execute(collected_data, feature_name=feature_name, **kwargs)
            return features
        else:
            logger.warning("[DataCollectionPipeline] Feature Pipeline not available, returning collected data")
            return collected_data
