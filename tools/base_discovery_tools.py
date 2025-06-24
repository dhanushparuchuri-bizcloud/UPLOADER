"""
Base discovery tools for data-driven pattern analysis
Uses statistical analysis instead of hardcoded assumptions
"""

import os
import json
import logging
import boto3
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from crewai.tools import BaseTool
import weaviate
import weaviate.classes.query as wq
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaDiscoveryMixin:
    """Mixin class for data-driven schema discovery"""
    
    def __init__(self):
        self.athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.s3_results_location = os.getenv('ATHENA_RESULTS_BUCKET', 's3://your-athena-results-bucket/')
        
    def execute_athena_query(self, query: str) -> List[Dict]:
        """Execute Athena query and return results"""
        try:
            response = self.athena_client.start_query_execution(
                QueryString=query,
                ResultConfiguration={'OutputLocation': self.s3_results_location},
                WorkGroup=os.getenv('ATHENA_WORKGROUP', 'primary')
            )
            
            query_id = response['QueryExecutionId']
            
            # Wait for query completion
            max_attempts = 30
            for attempt in range(max_attempts):
                result = self.athena_client.get_query_execution(QueryExecutionId=query_id)
                status = result['QueryExecution']['Status']['State']
                
                if status == 'SUCCEEDED':
                    break
                elif status in ['FAILED', 'CANCELLED']:
                    error_msg = result['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                    logger.error(f"Athena query failed: {error_msg}")
                    return []
                    
                import time
                time.sleep(2)
            
            # Get query results
            results = self.athena_client.get_query_results(QueryExecutionId=query_id)
            
            if not results.get('ResultSet', {}).get('Rows'):
                return []
            
            # Parse results
            columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            rows = []
            
            for row_data in results['ResultSet']['Rows'][1:]:  # Skip header row
                row = {}
                for i, cell in enumerate(row_data['Data']):
                    row[columns[i]] = cell.get('VarCharValue', '')
                rows.append(row)
            
            return rows
            
        except Exception as e:
            logger.error(f"Athena query execution failed: {e}")
            return []
    
    def discover_table_schema(self, table_name: str) -> Dict[str, Dict]:
        """Discover actual schema and infer business meaning from real data"""
        schema_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns 
        WHERE table_name = '{table_name.split('.')[-1]}'
        ORDER BY ordinal_position
        """
        
        columns = self.execute_athena_query(schema_query)
        
        if not columns:
            logger.warning(f"No schema information found for {table_name}")
            return {}
        
        # Analyze actual column names and types to infer purpose
        inferred_schema = {}
        for col in columns:
            col_name = col['column_name']
            data_type = col['data_type']
            
            # Get sample values and statistics
            sample_data = self._get_sample_values_and_stats(table_name, col_name, data_type)
            
            inferred_schema[col_name] = {
                "data_type": data_type,
                "is_nullable": col.get('is_nullable', 'YES'),
                "inferred_purpose": self._infer_column_purpose_from_data(
                    table_name, col_name, data_type, sample_data
                ),
                "sample_statistics": sample_data
            }
        
        return inferred_schema
    
    def _get_sample_values_and_stats(self, table_name: str, col_name: str, data_type: str) -> Dict:
        """Get sample values and basic statistics for a column"""
        try:
            # Basic stats query
            stats_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT {col_name}) as distinct_values,
                COUNT({col_name}) as non_null_values
            FROM {table_name}
            """
            
            stats_result = self.execute_athena_query(stats_query)
            if not stats_result:
                return {"error": "Could not retrieve statistics"}
            
            stats = stats_result[0]
            total_rows = int(stats.get('total_rows', 0))
            distinct_values = int(stats.get('distinct_values', 0))
            non_null_values = int(stats.get('non_null_values', 0))
            
            # Calculate cardinality ratio
            cardinality_ratio = distinct_values / total_rows if total_rows > 0 else 0
            null_ratio = (total_rows - non_null_values) / total_rows if total_rows > 0 else 0
            
            # Get sample values
            sample_query = f"""
            SELECT 
                {col_name},
                COUNT(*) as frequency
            FROM {table_name}
            WHERE {col_name} IS NOT NULL
            GROUP BY {col_name}
            ORDER BY frequency DESC
            LIMIT 20
            """
            
            sample_results = self.execute_athena_query(sample_query)
            sample_values = [row[col_name] for row in sample_results] if sample_results else []
            
            return {
                "total_rows": total_rows,
                "distinct_values": distinct_values,
                "non_null_values": non_null_values,
                "cardinality_ratio": cardinality_ratio,
                "null_ratio": null_ratio,
                "sample_values": sample_values[:10],  # Top 10 most frequent values
                "sample_count": len(sample_values)
            }
            
        except Exception as e:
            logger.warning(f"Could not get sample data for {col_name}: {e}")
            return {"error": str(e)}
    
    def _infer_column_purpose_from_data(self, table_name: str, col_name: str, data_type: str, sample_data: Dict) -> Dict:
        """Infer column purpose through statistical analysis of actual data"""
        
        if sample_data.get("error"):
            return {"category": "unknown", "confidence": "error", "error": sample_data["error"]}
        
        cardinality_ratio = sample_data.get("cardinality_ratio", 0)
        null_ratio = sample_data.get("null_ratio", 0)
        sample_values = sample_data.get("sample_values", [])
        total_rows = sample_data.get("total_rows", 0)
        
        # High cardinality analysis (likely identifiers or continuous metrics)
        if cardinality_ratio > 0.9:
            return self._analyze_high_cardinality_column(col_name, data_type, sample_values, sample_data)
        
        # Low cardinality analysis (likely categories or flags)
        elif cardinality_ratio < 0.1:
            return self._analyze_low_cardinality_column(col_name, data_type, sample_values, sample_data)
        
        # Medium cardinality analysis (dates, locations, moderate repetition)
        else:
            return self._analyze_medium_cardinality_column(col_name, data_type, sample_values, sample_data)
    
    def _analyze_high_cardinality_column(self, col_name: str, data_type: str, sample_values: List, stats: Dict) -> Dict:
        """Analyze columns with high cardinality (likely identifiers or continuous metrics)"""
        
        # Check for sequential ID patterns
        if data_type.lower() in ['bigint', 'integer', 'int']:
            try:
                numeric_values = [int(val) for val in sample_values if str(val).isdigit()]
                if len(numeric_values) > 3:
                    sorted_vals = sorted(numeric_values)
                    gaps = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1) if i < len(sorted_vals)-1]
                    avg_gap = sum(gaps) / len(gaps) if gaps else float('inf')
                    
                    if avg_gap < 100:  # Small gaps suggest sequential IDs
                        return {
                            "category": "identifier", 
                            "confidence": "high", 
                            "pattern": "sequential_id",
                            "evidence": f"Sequential numeric pattern with avg gap {avg_gap:.1f}"
                        }
            except (ValueError, TypeError):
                pass
        
        # Check for UUID patterns
        uuid_like = any(
            isinstance(val, str) and len(str(val)) == 36 and str(val).count('-') == 4 
            for val in sample_values
        )
        if uuid_like:
            return {
                "category": "identifier", 
                "confidence": "high", 
                "pattern": "uuid",
                "evidence": "UUID-like string patterns detected"
            }
        
        # Check for financial patterns (continuous numeric with reasonable ranges)
        if data_type.lower() in ['double', 'decimal', 'float']:
            try:
                numeric_values = [float(val) for val in sample_values if val is not None]
                if numeric_values:
                    positive_ratio = sum(1 for val in numeric_values if val > 0) / len(numeric_values)
                    max_val = max(numeric_values)
                    min_val = min(numeric_values)
                    
                    if positive_ratio > 0.8 and max_val < 1000000 and min_val >= 0:
                        return {
                            "category": "financial", 
                            "confidence": "medium", 
                            "pattern": "continuous_metric",
                            "evidence": f"Mostly positive numeric values, range: {min_val:.2f} to {max_val:.2f}"
                        }
            except (ValueError, TypeError):
                pass
        
        return {
            "category": "unknown", 
            "confidence": "low", 
            "pattern": "high_cardinality",
            "evidence": f"High cardinality ({stats.get('cardinality_ratio', 0):.2f}) but no clear pattern"
        }
    
    def _analyze_low_cardinality_column(self, col_name: str, data_type: str, sample_values: List, stats: Dict) -> Dict:
        """Analyze columns with low cardinality (likely categories or flags)"""
        
        unique_values = list(set(str(val).lower() for val in sample_values if val is not None))
        value_count = len(unique_values)
        
        # Boolean-like patterns
        if value_count == 2:
            boolean_patterns = [
                ({'0', '1'}, 'boolean_flag'),
                ({'true', 'false'}, 'boolean_flag'),
                ({'yes', 'no'}, 'boolean_flag'),
                ({'active', 'inactive'}, 'status_flag'),
                ({'enabled', 'disabled'}, 'status_flag'),
            ]
            
            for pattern_values, category in boolean_patterns:
                if set(unique_values) == pattern_values:
                    return {
                        "category": "categorical", 
                        "confidence": "high", 
                        "pattern": category,
                        "evidence": f"Binary values: {unique_values}"
                    }
        
        # Status/workflow patterns (3-10 values)
        elif 3 <= value_count <= 10:
            status_keywords = ['pending', 'complete', 'cancelled', 'in_progress', 'failed', 'success', 'draft']
            if any(keyword in ' '.join(unique_values) for keyword in status_keywords):
                return {
                    "category": "status", 
                    "confidence": "high", 
                    "pattern": "workflow_status",
                    "evidence": f"Status-like values: {unique_values}"
                }
            
            # Geographic patterns (state codes, regions)
            if all(len(str(val)) == 2 for val in sample_values if val is not None):
                return {
                    "category": "geographic", 
                    "confidence": "medium", 
                    "pattern": "state_code",
                    "evidence": f"Two-character codes: {unique_values}"
                }
        
        return {
            "category": "categorical", 
            "confidence": "medium", 
            "pattern": f"low_cardinality_{value_count}_values",
            "evidence": f"Categorical with {value_count} unique values: {unique_values}"
        }
    
    def _analyze_medium_cardinality_column(self, col_name: str, data_type: str, sample_values: List, stats: Dict) -> Dict:
        """Analyze columns with medium cardinality"""
        
        # Check for date patterns
        import re
        date_patterns = [
            (r'\d{4}-\d{2}-\d{2}', 'date_iso'),
            (r'\d{2}/\d{2}/\d{4}', 'date_us'),
            (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'timestamp'),
        ]
        
        for pattern, date_type in date_patterns:
            if any(re.match(pattern, str(val)) for val in sample_values if val is not None):
                return {
                    "category": "temporal", 
                    "confidence": "high", 
                    "pattern": date_type,
                    "evidence": f"Date/time pattern detected: {date_type}"
                }
        
        # Check for geographic patterns (zip codes)
        if all(str(val).isdigit() and len(str(val)) == 5 for val in sample_values if val is not None):
            return {
                "category": "geographic", 
                "confidence": "high", 
                "pattern": "zip_code",
                "evidence": "5-digit numeric codes consistent with zip codes"
            }
        
        # Check for financial patterns
        if data_type.lower() in ['double', 'decimal', 'float']:
            try:
                numeric_values = [float(val) for val in sample_values if val is not None]
                if numeric_values:
                    positive_ratio = sum(1 for val in numeric_values if val > 0) / len(numeric_values)
                    
                    if positive_ratio > 0.8:
                        return {
                            "category": "financial", 
                            "confidence": "medium", 
                            "pattern": "currency_amount",
                            "evidence": f"Mostly positive numeric values, {positive_ratio:.1%} positive"
                        }
            except (ValueError, TypeError):
                pass
        
        cardinality_ratio = stats.get('cardinality_ratio', 0)
        return {
            "category": "unknown", 
            "confidence": "low", 
            "pattern": f"medium_cardinality",
            "evidence": f"Medium cardinality ({cardinality_ratio:.2f}) without clear pattern"
        }