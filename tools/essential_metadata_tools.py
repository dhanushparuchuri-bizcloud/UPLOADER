# tools/essential_metadata_tools.py
"""
Essential tools for moving services metadata generation using AWS Bedrock
"""

import os
import json
import logging
import boto3
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from tools.weaviate_client import WeaviateClientSingleton
import weaviate.classes.query as wq
import re

logger = logging.getLogger(__name__)

class AthenaSamplingTool(BaseTool):
    name: str = "Athena Data Sampler"
    description: str = "Extract real sample data and basic facts from moving services tables using AWS Athena"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, table_name: str, sample_size: int = 100) -> str:
        """Get real data samples and basic statistics from the table"""
        try:
            logger.info(f"🔍 Sampling {sample_size} rows from {table_name}")
            
            # Initialize AWS clients in the method
            athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
            s3_results_location = os.getenv('ATHENA_RESULTS_BUCKET', 's3://amspoc3queryresults/')
            athena_workgroup = os.getenv('ATHENA_WORKGROUP', 'primary')
            athena_database = os.getenv('ATHENA_DATABASE', 'amspoc3test')
            
            # Ensure table name includes database if not already specified
            if '.' not in table_name:
                full_table_name = f"{athena_database}.{table_name}"
            else:
                full_table_name = table_name
            
            # Get sample data
            sample_data = self._get_sample_data(full_table_name, sample_size, athena_client, s3_results_location, athena_workgroup)
            if not sample_data:
                return json.dumps({
                    "error": f"No data found in {full_table_name}",
                    "table_name": table_name,
                    "status": "empty_table"
                })
            
            # Get table schema information
            table_schema = self._get_table_schema(full_table_name, athena_client, s3_results_location, athena_workgroup, athena_database)
            
            # Analyze each column
            column_analysis = {}
            if sample_data and len(sample_data) > 0:
                # Get column names from first row
                column_names = list(sample_data[0].keys()) if sample_data else []
                
                for column_name in column_names:
                    column_analysis[column_name] = self._analyze_column_samples(
                        sample_data, column_name, table_schema.get(column_name, {})
                    )
            
            result = {
                "table_name": table_name,
                "full_table_name": full_table_name,
                "sample_size_requested": sample_size,
                "sample_size_actual": len(sample_data),
                "columns_found": len(column_analysis),
                "table_schema": table_schema,
                "column_analysis": column_analysis,
                "sample_preview": sample_data[:5] if sample_data else [],
                "status": "success"
            }
            
            logger.info(f"✅ Successfully sampled {len(sample_data)} rows with {len(column_analysis)} columns")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"❌ Athena sampling failed: {e}")
            return json.dumps({
                "error": str(e),
                "table_name": table_name,
                "status": "failed"
            })
    
    def _execute_athena_query(self, query: str, athena_client, s3_results_location: str, athena_workgroup: str) -> List[Dict]:
        """Execute Athena query and return results"""
        try:
            response = athena_client.start_query_execution(
                QueryString=query,
                ResultConfiguration={'OutputLocation': s3_results_location},
                WorkGroup=athena_workgroup
            )
            
            query_id = response['QueryExecutionId']
            
            # Wait for query completion
            max_attempts = 30
            for attempt in range(max_attempts):
                result = athena_client.get_query_execution(QueryExecutionId=query_id)
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
            results = athena_client.get_query_results(QueryExecutionId=query_id)
            
            if not results.get('ResultSet', {}).get('Rows'):
                return []
            
            # Parse results
            columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            rows = []
            
            for row_data in results['ResultSet']['Rows'][1:]:  # Skip header row
                row = {}
                for i, cell in enumerate(row_data['Data']):
                    value = cell.get('VarCharValue', '')
                    # Try to convert to appropriate type
                    if value == '':
                        row[columns[i]] = None
                    elif value.isdigit():
                        row[columns[i]] = int(value)
                    else:
                        try:
                            row[columns[i]] = float(value)
                        except ValueError:
                            row[columns[i]] = value
                rows.append(row)
            
            return rows
            
        except Exception as e:
            logger.error(f"Athena query execution failed: {e}")
            return []
    
    def _get_sample_data(self, table_name: str, sample_size: int, athena_client, s3_results_location: str, athena_workgroup: str) -> List[Dict]:
        """Get random sample of data from the table"""
        sample_query = f"""
        SELECT *
        FROM {table_name}
        TABLESAMPLE BERNOULLI(5)
        LIMIT {sample_size}
        """
        
        return self._execute_athena_query(sample_query, athena_client, s3_results_location, athena_workgroup)
    
    def _get_table_schema(self, table_name: str, athena_client, s3_results_location: str, athena_workgroup: str, athena_database: str) -> Dict[str, Dict]:
        """Get basic schema information"""
        try:
            # Extract database and table name
            if '.' in table_name:
                database, table = table_name.split('.', 1)
            else:
                database = athena_database
                table = table_name
            
            schema_query = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = '{database}' 
            AND table_name = '{table}'
            ORDER BY ordinal_position
            """
            
            schema_results = self._execute_athena_query(schema_query, athena_client, s3_results_location, athena_workgroup)
            
            schema = {}
            for row in schema_results:
                col_name = row.get('column_name', '')
                schema[col_name] = {
                    'data_type': row.get('data_type', 'unknown'),
                    'is_nullable': row.get('is_nullable', 'YES')
                }
            
            return schema
            
        except Exception as e:
            logger.warning(f"Could not get schema for {table_name}: {e}")
            return {}
    
    def _analyze_column_samples(self, sample_data: List[Dict], column_name: str, schema_info: Dict) -> Dict:
        """Analyze sample values for a specific column"""
        values = [row.get(column_name) for row in sample_data]
        non_null_values = [v for v in values if v is not None and v != '']
        
        if not non_null_values:
            return {
                "data_type": schema_info.get('data_type', 'unknown'),
                "sample_values": [],
                "total_samples": len(values),
                "non_null_samples": 0,
                "null_percentage": 100.0,
                "distinct_values": 0,
                "analysis": "All values are null or empty"
            }
        
        # Get unique values for analysis
        unique_values = list(set(non_null_values))
        
        # Sample values for metadata (limit to prevent oversized output)
        sample_values = unique_values[:20] if len(unique_values) <= 20 else unique_values[:10]
        
        analysis = {
            "data_type": schema_info.get('data_type', 'unknown'),
            "sample_values": sample_values,
            "total_samples": len(values),
            "non_null_samples": len(non_null_values),
            "null_percentage": round((len(values) - len(non_null_values)) / len(values) * 100, 2),
            "distinct_values": len(unique_values),
            "cardinality_ratio": round(len(unique_values) / len(non_null_values), 4) if non_null_values else 0
        }
        
        # Add pattern analysis
        analysis["pattern_analysis"] = self._detect_value_patterns(non_null_values, unique_values)
        
        return analysis
    
    def _detect_value_patterns(self, values: List, unique_values: List) -> Dict:
        """Detect patterns in the values"""
        patterns = {
            "value_type": "mixed",
            "has_numeric": False,
            "has_dates": False,
            "has_codes": False,
            "typical_length": 0
        }
        
        if not values:
            return patterns
        
        # Check for numeric values
        numeric_count = sum(1 for v in values if isinstance(v, (int, float)))
        patterns["has_numeric"] = numeric_count > len(values) * 0.8
        
        # Check for date patterns
        string_values = [str(v) for v in values if v is not None]
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'  # YYYY-MM-DD HH:MM:SS
        ]
        
        for pattern in date_patterns:
            date_matches = sum(1 for v in string_values if re.match(pattern, str(v)))
            if date_matches > len(string_values) * 0.7:
                patterns["has_dates"] = True
                break
        
        # Check for code patterns (short alphanumeric)
        if string_values:
            avg_length = sum(len(str(v)) for v in string_values) / len(string_values)
            patterns["typical_length"] = round(avg_length, 1)
            
            if 2 <= avg_length <= 10 and len(unique_values) < len(values) * 0.3:
                patterns["has_codes"] = True
        
        # Determine overall type
        if patterns["has_numeric"]:
            patterns["value_type"] = "numeric"
        elif patterns["has_dates"]:
            patterns["value_type"] = "temporal"
        elif patterns["has_codes"]:
            patterns["value_type"] = "categorical_code"
        elif len(unique_values) <= 10:
            patterns["value_type"] = "categorical"
        else:
            patterns["value_type"] = "text"
        
        return patterns


class ColumnProfilerTool(BaseTool):
    name: str = "Column Statistical Profiler"
    description: str = "Get detailed statistical context for moving services table columns using AWS Athena"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, table_name: str, column_name: str) -> str:
        """Get detailed statistical profile for a specific column"""
        try:
            logger.info(f"📊 Profiling column {column_name} in {table_name}")
            
            # Initialize AWS clients in the method
            athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
            s3_results_location = os.getenv('ATHENA_RESULTS_BUCKET', 's3://amspoc3queryresults/')
            athena_workgroup = os.getenv('ATHENA_WORKGROUP', 'primary')
            athena_database = os.getenv('ATHENA_DATABASE', 'amspoc3test')
            
            # Ensure table name includes database
            if '.' not in table_name:
                full_table_name = f"{athena_database}.{table_name}"
            else:
                full_table_name = table_name
            
            # Get comprehensive statistics
            stats = self._get_column_statistics(full_table_name, column_name, athena_client, s3_results_location, athena_workgroup)
            
            if stats.get("error"):
                return json.dumps({
                    "error": stats["error"],
                    "table_name": table_name,
                    "column_name": column_name,
                    "status": "failed"
                })
            
            # Get value distribution
            distribution = self._get_value_distribution(full_table_name, column_name, athena_client, s3_results_location, athena_workgroup)
            
            result = {
                "table_name": table_name,
                "column_name": column_name,
                "statistics": stats,
                "value_distribution": distribution,
                "business_insights": self._generate_business_insights(stats, distribution, column_name),
                "status": "success"
            }
            
            logger.info(f"✅ Successfully profiled {column_name}")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"❌ Column profiling failed: {e}")
            return json.dumps({
                "error": str(e),
                "table_name": table_name,
                "column_name": column_name,
                "status": "failed"
            })
    
    def _execute_athena_query(self, query: str, athena_client, s3_results_location: str, athena_workgroup: str) -> List[Dict]:
        """Execute Athena query and return results"""
        try:
            response = athena_client.start_query_execution(
                QueryString=query,
                ResultConfiguration={'OutputLocation': s3_results_location},
                WorkGroup=athena_workgroup
            )
            
            query_id = response['QueryExecutionId']
            
            # Wait for query completion
            max_attempts = 30
            for attempt in range(max_attempts):
                result = athena_client.get_query_execution(QueryExecutionId=query_id)
                status = result['QueryExecution']['Status']['State']
                
                if status == 'SUCCEEDED':
                    break
                elif status in ['FAILED', 'CANCELLED']:
                    error_msg = result['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                    return [{"error": error_msg}]
                    
                import time
                time.sleep(1)
            
            # Get query results
            results = athena_client.get_query_results(QueryExecutionId=query_id)
            
            if not results.get('ResultSet', {}).get('Rows'):
                return []
            
            # Parse results
            columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            rows = []
            
            for row_data in results['ResultSet']['Rows'][1:]:  # Skip header row
                row = {}
                for i, cell in enumerate(row_data['Data']):
                    value = cell.get('VarCharValue', '')
                    # Convert to appropriate type
                    if value == '' or value is None:
                        row[columns[i]] = None
                    elif value.isdigit():
                        row[columns[i]] = int(value)
                    else:
                        try:
                            row[columns[i]] = float(value)
                        except ValueError:
                            row[columns[i]] = value
                rows.append(row)
            
            return rows
            
        except Exception as e:
            logger.error(f"Athena query execution failed: {e}")
            return [{"error": str(e)}]
    
    def _get_column_statistics(self, table_name: str, column_name: str, athena_client, s3_results_location: str, athena_workgroup: str) -> Dict:
        """Get comprehensive statistics for the column"""
        stats_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT({column_name}) as non_null_count,
            COUNT(DISTINCT {column_name}) as distinct_count,
            CAST(COUNT({column_name}) AS DOUBLE) / COUNT(*) as non_null_ratio,
            CAST(COUNT(DISTINCT {column_name}) AS DOUBLE) / COUNT({column_name}) as cardinality_ratio
        FROM {table_name}
        """
        
        results = self._execute_athena_query(stats_query, athena_client, s3_results_location, athena_workgroup)
        
        if not results or results[0].get("error"):
            return {"error": results[0].get("error", "Statistics query failed") if results else "No results"}
        
        stats = results[0]
        return {
            "total_rows": stats.get("total_rows", 0),
            "non_null_count": stats.get("non_null_count", 0),
            "distinct_count": stats.get("distinct_count", 0),
            "null_count": stats.get("total_rows", 0) - stats.get("non_null_count", 0),
            "null_percentage": round((1 - stats.get("non_null_ratio", 0)) * 100, 2),
            "cardinality_ratio": round(stats.get("cardinality_ratio", 0), 4),
            "non_null_ratio": round(stats.get("non_null_ratio", 0), 4)
        }
    
    def _get_value_distribution(self, table_name: str, column_name: str, athena_client, s3_results_location: str, athena_workgroup: str, limit: int = 20) -> Dict:
        """Get value distribution for the column"""
        distribution_query = f"""
        SELECT 
            {column_name} as value,
            COUNT(*) as frequency,
            CAST(COUNT(*) AS DOUBLE) / (SELECT COUNT(*) FROM {table_name}) as percentage
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        GROUP BY {column_name}
        ORDER BY frequency DESC
        LIMIT {limit}
        """
        
        results = self._execute_athena_query(distribution_query, athena_client, s3_results_location, athena_workgroup)
        
        if not results or (results and results[0].get("error")):
            return {"error": results[0].get("error", "Distribution query failed") if results else "No results"}
        
        distribution = {
            "top_values": [],
            "value_count": len(results),
            "top_values_coverage": 0.0
        }
        
        total_percentage = 0.0
        for row in results:
            value_info = {
                "value": row.get("value"),
                "frequency": row.get("frequency", 0),
                "percentage": round(row.get("percentage", 0) * 100, 2)
            }
            distribution["top_values"].append(value_info)
            total_percentage += value_info["percentage"]
        
        distribution["top_values_coverage"] = round(total_percentage, 2)
        
        return distribution
    
    def _generate_business_insights(self, stats: Dict, distribution: Dict, column_name: str) -> Dict:
        """Generate business insights based on statistical patterns"""
        insights = {
            "data_quality": "good",
            "business_pattern": "unknown",
            "analytical_utility": "medium",
            "recommendations": []
        }
        
        if stats.get("error") or distribution.get("error"):
            insights["data_quality"] = "analysis_failed"
            return insights
        
        # Data quality assessment
        null_percentage = stats.get("null_percentage", 0)
        if null_percentage > 50:
            insights["data_quality"] = "poor"
            insights["recommendations"].append("High null percentage - investigate data completeness")
        elif null_percentage > 20:
            insights["data_quality"] = "fair"
            insights["recommendations"].append("Moderate null percentage - consider data validation")
        
        # Business pattern detection
        cardinality_ratio = stats.get("cardinality_ratio", 0)
        distinct_count = stats.get("distinct_count", 0)
        
        if cardinality_ratio > 0.95:
            insights["business_pattern"] = "identifier"
            insights["analytical_utility"] = "high"
            insights["recommendations"].append("High cardinality suggests unique identifier - useful for joins")
        elif cardinality_ratio < 0.1 and distinct_count <= 20:
            insights["business_pattern"] = "categorical"
            insights["analytical_utility"] = "high"
            insights["recommendations"].append("Low cardinality suggests categorical data - good for grouping and filtering")
        elif distinct_count <= 10:
            insights["business_pattern"] = "status_or_type"
            insights["analytical_utility"] = "high"
            insights["recommendations"].append("Limited distinct values suggest status or type field")
        
        # Coverage analysis
        if distribution.get("top_values_coverage", 0) > 80:
            insights["recommendations"].append("Top values represent most data - good for analysis")
        
        # Moving services specific insights
        column_lower = column_name.lower()
        if any(term in column_lower for term in ['crew', 'team', 'worker']):
            insights["moving_services_context"] = "crew_operations"
        elif any(term in column_lower for term in ['rate', 'cost', 'price', 'revenue']):
            insights["moving_services_context"] = "financial_metrics"
        elif any(term in column_lower for term in ['location', 'branch', 'city', 'state']):
            insights["moving_services_context"] = "geographic_operations"
        elif any(term in column_lower for term in ['status', 'stage', 'step']):
            insights["moving_services_context"] = "workflow_tracking"
        
        return insights


class NameDeconstructionTool(BaseTool):
    name: str = "Technical Name Deconstructor"
    description: str = "Convert technical database column names to business-friendly names using moving services terminology"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, column_name: str, context: str = "") -> str:
        """Convert technical column name to business-friendly name"""
        try:
            logger.info(f"🏷️ Deconstructing column name: {column_name}")
            
            # Initialize Bedrock client in the method
            bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
            
            # Quick rule-based conversion for common patterns
            rule_based_result = self._rule_based_conversion(column_name)
            
            # Use AI for more sophisticated conversion
            ai_result = self._ai_enhanced_conversion(column_name, context, rule_based_result, bedrock_client, model_id)
            
            result = {
                "technical_name": column_name,
                "business_name": ai_result.get("business_name", rule_based_result),
                "confidence": ai_result.get("confidence", "medium"),
                "reasoning": ai_result.get("reasoning", "Rule-based conversion"),
                "alternatives": ai_result.get("alternatives", []),
                "moving_services_context": ai_result.get("moving_services_context", ""),
                "status": "success"
            }
            
            logger.info(f"✅ Converted '{column_name}' to '{result['business_name']}'")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Name deconstruction failed: {e}")
            return json.dumps({
                "technical_name": column_name,
                "business_name": self._rule_based_conversion(column_name),
                "error": str(e),
                "status": "fallback"
            })
    
    def _rule_based_conversion(self, column_name: str) -> str:
        """Simple rule-based conversion for common patterns"""
        # Common abbreviation mappings for moving services
        abbreviations = {
            'id': 'ID',
            'num': 'Number',
            'qty': 'Quantity',
            'amt': 'Amount',
            'dt': 'Date',
            'tm': 'Time',
            'addr': 'Address',
            'st': 'State',
            'zip': 'ZIP Code',
            'cd': 'Code',
            'desc': 'Description',
            'cat': 'Category',
            'typ': 'Type',
            'stat': 'Status',
            'crew': 'Crew',
            'emp': 'Employee',
            'cust': 'Customer',
            'svc': 'Service',
            'rev': 'Revenue',
            'cost': 'Cost',
            'rate': 'Rate',
            'hrs': 'Hours',
            'mins': 'Minutes',
            'loc': 'Location',
            'br': 'Branch',
            'reg': 'Region'
        }
        
        # Split camelCase and snake_case
        if '_' in column_name:
            parts = column_name.split('_')
        else:
            # Split camelCase
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', column_name)
            if not parts:
                parts = [column_name]
        
        # Convert each part
        converted_parts = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in abbreviations:
                converted_parts.append(abbreviations[part_lower])
            else:
                converted_parts.append(part.capitalize())
        
        return ' '.join(converted_parts)
    
    def _ai_enhanced_conversion(self, column_name: str, context: str, rule_based: str, bedrock_client, model_id: str) -> Dict:
        """Use Bedrock AI for enhanced name conversion"""
        try:
            prompt = f"""Convert the technical database column name to a business-friendly name for a moving services company.

Technical Column Name: {column_name}
Rule-based Suggestion: {rule_based}
Additional Context: {context}

Moving Services Business Context:
- This is a national moving company with crew-based operations
- Services include residential, commercial, and corporate moves
- Key business areas: crew management, service delivery, geographic operations, financial tracking
- Service tiers: Full Packing, Partial Packing, Owner Packing
- Operational metrics: labor hours, completion times, efficiency measures

Requirements:
- Business name should be clear to operations managers, finance teams, and executives
- Use terminology familiar to the moving industry
- Keep it concise but descriptive
- Consider the business context and likely usage

Respond with JSON only:
{{
  "business_name": "Clear Business Name",
  "confidence": "high|medium|low", 
  "reasoning": "Why this name was chosen",
  "alternatives": ["Alternative 1", "Alternative 2"],
  "moving_services_context": "How this relates to moving operations"
}}"""

            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "temperature": 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
                return ai_result
            else:
                # Fallback if JSON parsing fails
                return {
                    "business_name": rule_based,
                    "confidence": "low",
                    "reasoning": "AI parsing failed, used rule-based conversion"
                }
                
        except Exception as e:
            logger.warning(f"AI name conversion failed: {e}")
            return {
                "business_name": rule_based,
                "confidence": "medium", 
                "reasoning": f"AI conversion failed ({str(e)}), used rule-based conversion"
            }