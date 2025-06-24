"""
Data-driven pattern discovery tools for moving services metadata generation
"""

import json
import logging
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from .base_discovery_tools import SchemaDiscoveryMixin
from .weaviate_client import WeaviateClientSingleton
import weaviate.classes.query as wq
import os
import boto3
import re
from statistics import correlation, StatisticsError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AthenaSamplingTool(BaseTool, SchemaDiscoveryMixin):
    name: str = "Athena Data Sampler"
    description: str = "Sample actual data from tables to understand patterns and business context through statistical analysis"
    
    # Declare Pydantic fields for the boto3 clients and configuration
    athena_client: Any = None
    s3_results_location: str = ""
    
    def __init__(self):
        # Initialize the base tool first
        super().__init__()
        # Then set up the AWS clients
        self.athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.s3_results_location = os.getenv('ATHENA_RESULTS_BUCKET', 's3://your-athena-results-bucket/')
    
    def _run(self, table_name: str, column_name: str = None) -> str:
        """Sample data and analyze patterns for moving services business context"""
        try:
            logger.info(f"🔍 Sampling data from {table_name}")
            
            # Discover schema first
            schema = self.discover_table_schema(table_name)
            
            if not schema:
                return json.dumps({
                    "error": f"Could not discover schema for {table_name}",
                    "analysis": "No data available for pattern analysis"
                })
            
            # Overall table analysis
            table_analysis = self._analyze_table_overview(table_name, schema)
            
            # Column-specific analysis if requested
            column_analysis = {}
            if column_name and column_name in schema:
                column_analysis = self._analyze_specific_column(table_name, column_name, schema[column_name])
            
            # Business pattern insights
            business_insights = self._generate_business_insights(table_name, schema, table_analysis)
            
            return json.dumps({
                "table_name": table_name,
                "schema_discovered": len(schema),
                "table_analysis": table_analysis,
                "column_analysis": column_analysis,
                "business_insights": business_insights,
                "schema_summary": {col: info["inferred_purpose"] for col, info in schema.items()}
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Athena sampling failed: {e}")
            return json.dumps({"error": str(e), "analysis": "Sampling failed"})
    
    def _analyze_table_overview(self, table_name: str, schema: Dict) -> Dict:
        """Analyze overall table characteristics"""
        try:
            overview_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT *) as unique_records
            FROM {table_name}
            """
            
            results = self.execute_athena_query(overview_query)
            if not results:
                return {"error": "Could not retrieve table overview"}
            
            total_records = int(results[0].get('total_records', 0))
            unique_records = int(results[0].get('unique_records', 0))
            
            # Analyze column types
            column_types = {}
            for col_name, col_info in schema.items():
                purpose = col_info["inferred_purpose"]
                category = purpose.get("category", "unknown")
                column_types[category] = column_types.get(category, 0) + 1
            
            return {
                "total_records": total_records,
                "unique_records": unique_records,
                "duplicate_ratio": (total_records - unique_records) / total_records if total_records > 0 else 0,
                "column_types_distribution": column_types,
                "data_quality_indicators": self._assess_data_quality(schema)
            }
            
        except Exception as e:
            logger.warning(f"Table overview analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_specific_column(self, table_name: str, column_name: str, column_info: Dict) -> Dict:
        """Detailed analysis of a specific column"""
        try:
            sample_stats = column_info.get("sample_statistics", {})
            inferred_purpose = column_info.get("inferred_purpose", {})
            
            analysis = {
                "data_type": column_info.get("data_type"),
                "inferred_purpose": inferred_purpose,
                "sample_statistics": sample_stats,
                "business_implications": self._interpret_business_implications(
                    column_name, column_info
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Column analysis failed for {column_name}: {e}")
            return {"error": str(e)}
    
    def _assess_data_quality(self, schema: Dict) -> Dict:
        """Assess data quality indicators"""
        quality_indicators = {
            "high_null_columns": [],
            "low_cardinality_columns": [],
            "potential_identifiers": [],
            "data_quality_score": 0.0
        }
        
        total_columns = len(schema)
        quality_score = 0
        
        for col_name, col_info in schema.items():
            stats = col_info.get("sample_statistics", {})
            purpose = col_info.get("inferred_purpose", {})
            
            null_ratio = stats.get("null_ratio", 0)
            cardinality_ratio = stats.get("cardinality_ratio", 0)
            
            # Check for data quality issues
            if null_ratio > 0.5:
                quality_indicators["high_null_columns"].append({
                    "column": col_name,
                    "null_ratio": null_ratio
                })
            else:
                quality_score += 1
            
            if cardinality_ratio < 0.01 and purpose.get("category") != "categorical":
                quality_indicators["low_cardinality_columns"].append({
                    "column": col_name,
                    "cardinality_ratio": cardinality_ratio
                })
            
            if purpose.get("category") == "identifier":
                quality_indicators["potential_identifiers"].append({
                    "column": col_name,
                    "pattern": purpose.get("pattern", "unknown")
                })
                quality_score += 0.5  # Identifiers are good for joins
        
        quality_indicators["data_quality_score"] = quality_score / total_columns if total_columns > 0 else 0
        
        return quality_indicators
    
    def _interpret_business_implications(self, column_name: str, column_info: Dict) -> Dict:
        """Interpret business implications based on inferred purpose"""
        purpose = column_info.get("inferred_purpose", {})
        category = purpose.get("category", "unknown")
        pattern = purpose.get("pattern", "")
        
        implications = {
            "operational_impact": "Unknown",
            "financial_impact": "Unknown", 
            "analytical_potential": "Unknown"
        }
        
        if category == "identifier":
            implications.update({
                "operational_impact": "Can be used for joining with other datasets",
                "financial_impact": "No direct financial impact",
                "analytical_potential": "High - enables data relationships and tracking"
            })
        
        elif category == "financial":
            implications.update({
                "operational_impact": "May represent costs or revenue affecting operations",
                "financial_impact": "Direct impact on financial calculations and reporting",
                "analytical_potential": "High - can be aggregated for profitability analysis"
            })
        
        elif category == "temporal":
            implications.update({
                "operational_impact": "Enables time-based analysis and scheduling optimization",
                "financial_impact": "Important for period-based financial reporting",
                "analytical_potential": "High - enables trend and seasonal analysis"
            })
        
        elif category == "geographic":
            implications.update({
                "operational_impact": "Critical for regional operations and logistics planning",
                "financial_impact": "Affects regional cost structures and pricing",
                "analytical_potential": "High - enables geographic performance comparison"
            })
        
        elif category == "categorical" or category == "status":
            implications.update({
                "operational_impact": "Useful for filtering and operational state tracking",
                "financial_impact": "May represent different cost or revenue categories",
                "analytical_potential": "Medium - useful for segmentation and classification"
            })
        
        return implications
    
    def _generate_business_insights(self, table_name: str, schema: Dict, table_analysis: Dict) -> Dict:
        """Generate business insights for moving services context"""
        insights = {
            "moving_services_indicators": [],
            "operational_columns": [],
            "financial_columns": [],
            "geographic_columns": [],
            "temporal_columns": [],
            "recommended_analysis": []
        }
        
        # Analyze for moving services patterns
        for col_name, col_info in schema.items():
            purpose = col_info.get("inferred_purpose", {})
            category = purpose.get("category", "unknown")
            
            # Categorize by business function
            if category == "financial":
                insights["financial_columns"].append({
                    "column": col_name,
                    "business_use": "Cost analysis, revenue tracking, profitability calculations"
                })
            
            elif category == "geographic":
                insights["geographic_columns"].append({
                    "column": col_name,
                    "business_use": "Regional performance analysis, market comparison"
                })
            
            elif category == "temporal":
                insights["temporal_columns"].append({
                    "column": col_name,
                    "business_use": "Seasonal analysis, trend identification, scheduling optimization"
                })
            
            elif category in ["categorical", "status"]:
                insights["operational_columns"].append({
                    "column": col_name,
                    "business_use": "Operational filtering, status tracking, performance segmentation"
                })
        
        # Generate recommendations
        if insights["financial_columns"]:
            insights["recommended_analysis"].append(
                "Analyze financial columns for profitability patterns and cost optimization"
            )
        
        if insights["geographic_columns"]:
            insights["recommended_analysis"].append(
                "Compare geographic variations to identify market opportunities"
            )
        
        if insights["temporal_columns"]:
            insights["recommended_analysis"].append(
                "Analyze seasonal patterns to optimize resource planning"
            )
        
        # Look for moving services specific patterns
        moving_keywords = ['crew', 'truck', 'move', 'labor', 'branch', 'customer', 'revenue']
        for keyword in moving_keywords:
            matching_columns = [col for col in schema.keys() if keyword.lower() in col.lower()]
            if matching_columns:
                insights["moving_services_indicators"].append({
                    "keyword": keyword,
                    "columns": matching_columns,
                    "business_relevance": f"Indicates {keyword}-related business operations"
                })
        
        return insights


class CorrelationDiscoveryTool(BaseTool, SchemaDiscoveryMixin):
    name: str = "Statistical Business Correlation Discovery"
    description: str = "Discover actual data relationships through statistical analysis, not hardcoded patterns"
    
    # Declare Pydantic fields for the boto3 clients and configuration
    athena_client: Any = None
    s3_results_location: str = ""
    
    def __init__(self):
        # Initialize the base tool first
        super().__init__()
        # Then set up the AWS clients
        self.athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.s3_results_location = os.getenv('ATHENA_RESULTS_BUCKET', 's3://your-athena-results-bucket/')
    
    def _run(self, table_name: str, column_name: str) -> str:
        """Discover statistical correlations and business relationships"""
        try:
            logger.info(f"🔗 Discovering correlations for {column_name} in {table_name}")
            
            # Discover schema first
            schema = self.discover_table_schema(table_name)
            
            if column_name not in schema:
                return json.dumps({
                    "error": f"Column {column_name} not found in {table_name}",
                    "available_columns": list(schema.keys())
                })
            
            # Statistical correlation analysis
            correlations = self._find_statistical_correlations(table_name, column_name, schema)
            
            # Foreign key relationship discovery
            relationships = self._discover_fk_relationships(table_name, column_name, schema)
            
            # Business workflow analysis
            workflows = self._identify_business_workflows(table_name, column_name, schema)
            
            # Cross-column analysis
            cross_analysis = self._perform_cross_column_analysis(table_name, column_name, schema)
            
            return json.dumps({
                "target_column": column_name,
                "table_name": table_name,
                "statistical_correlations": correlations,
                "relationship_discovery": relationships,
                "business_workflows": workflows,
                "cross_column_analysis": cross_analysis,
                "analysis_summary": self._generate_correlation_summary(
                    correlations, relationships, workflows
                )
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Correlation discovery failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _find_statistical_correlations(self, table_name: str, target_column: str, schema: Dict) -> Dict:
        """Find statistically significant correlations with other numeric columns"""
        correlations = {}
        
        # Find numeric columns for correlation analysis
        numeric_columns = [
            col for col, info in schema.items()
            if info["data_type"].lower() in ['bigint', 'double', 'decimal', 'integer', 'float']
            and col != target_column
        ]
        
        if not numeric_columns:
            return {"message": "No numeric columns available for correlation analysis"}
        
        for other_col in numeric_columns[:5]:  # Limit to prevent too many queries
            try:
                corr_query = f"""
                SELECT 
                    CORR({target_column}, {other_col}) as correlation_coefficient,
                    COUNT(*) as sample_size
                FROM {table_name}
                WHERE {target_column} IS NOT NULL AND {other_col} IS NOT NULL
                """
                
                result = self.execute_athena_query(corr_query)
                if result and result[0]:
                    corr_coeff = float(result[0].get('correlation_coefficient', 0))
                    sample_size = int(result[0].get('sample_size', 0))
                    
                    # Only report significant correlations
                    if abs(corr_coeff) > 0.3 and sample_size > 30:
                        correlations[other_col] = {
                            "correlation": corr_coeff,
                            "strength": self._interpret_correlation_strength(corr_coeff),
                            "sample_size": sample_size,
                            "business_interpretation": self._interpret_correlation_business_meaning(
                                target_column, other_col, corr_coeff, schema
                            )
                        }
            
            except Exception as e:
                logger.warning(f"Correlation analysis failed for {other_col}: {e}")
        
        return correlations
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _interpret_correlation_business_meaning(self, col1: str, col2: str, correlation: float, schema: Dict) -> str:
        """Interpret business meaning of correlation"""
        col1_purpose = schema.get(col1, {}).get("inferred_purpose", {}).get("category", "unknown")
        col2_purpose = schema.get(col2, {}).get("inferred_purpose", {}).get("category", "unknown")
        
        direction = "positive" if correlation > 0 else "negative"
        
        if col1_purpose == "financial" and col2_purpose == "financial":
            return f"{direction.title()} relationship between cost/revenue factors - important for profitability analysis"
        
        elif col1_purpose == "financial" and col2_purpose == "identifier":
            return f"Financial metric varies by identifier - useful for segmented financial analysis"
        
        elif "temporal" in [col1_purpose, col2_purpose]:
            return f"Time-based relationship - indicates seasonal or trending patterns"
        
        else:
            return f"{direction.title()} operational relationship between {col1} and {col2}"
    
    def _discover_fk_relationships(self, table_name: str, column_name: str, schema: Dict) -> Dict:
        """Discover potential foreign key relationships through data analysis"""
        relationships = {"detected": False, "analysis": []}
        
        column_info = schema.get(column_name, {})
        inferred_purpose = column_info.get("inferred_purpose", {})
        
        # Only analyze if column looks like an identifier
        if inferred_purpose.get("category") != "identifier":
            relationships["analysis"].append(f"Column {column_name} doesn't appear to be an identifier")
            return relationships
        
        # Check for naming patterns that suggest foreign keys
        if column_name.endswith('_id') or column_name.endswith('ID'):
            potential_parent_table = column_name.replace('_id', '').replace('ID', '').lower()
            
            # Check if values exist in a potential parent table
            parent_check_query = f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE '%{potential_parent_table}%'
            """
            
            parent_tables = self.execute_athena_query(parent_check_query)
            
            if parent_tables:
                relationships["detected"] = True
                relationships["potential_parent_tables"] = [t["table_name"] for t in parent_tables]
                relationships["analysis"].append(
                    f"Found potential parent tables for {column_name}: {[t['table_name'] for t in parent_tables]}"
                )
            else:
                relationships["analysis"].append(f"No parent table found for pattern {potential_parent_table}")
        
        return relationships
    
    def _identify_business_workflows(self, table_name: str, column_name: str, schema: Dict) -> Dict:
        """Identify business workflows this column might participate in"""
        workflows = {}
        
        column_info = schema.get(column_name, {})
        inferred_purpose = column_info.get("inferred_purpose", {})
        category = inferred_purpose.get("category", "unknown")
        
        # Moving services specific workflow identification
        if category == "financial":
            workflows["profitability_analysis"] = {
                "related_columns": self._find_related_columns_by_category(schema, ["financial", "identifier"]),
                "business_purpose": "Calculate labor costs, revenue per move, and profit margins",
                "stakeholders": ["Finance Team", "Operations Manager"],
                "analysis_type": "profitability_calculation"
            }
        
        if category == "temporal":
            workflows["seasonal_analysis"] = {
                "related_columns": self._find_related_columns_by_category(schema, ["temporal", "financial"]),
                "business_purpose": "Analyze seasonal patterns affecting operations and revenue",
                "stakeholders": ["Operations Manager", "Business Leader"],
                "analysis_type": "trend_and_seasonality"
            }
        
        if category == "geographic":
            workflows["regional_performance"] = {
                "related_columns": self._find_related_columns_by_category(schema, ["geographic", "financial"]),
                "business_purpose": "Compare performance across different locations and markets",
                "stakeholders": ["Regional Manager", "Business Leader"],
                "analysis_type": "geographic_comparison"
            }
        
        return workflows
    
    def _find_related_columns_by_category(self, schema: Dict, categories: List[str]) -> List[str]:
        """Find columns that belong to specified categories"""
        related_columns = []
        for col_name, col_info in schema.items():
            purpose = col_info.get("inferred_purpose", {})
            if purpose.get("category") in categories:
                related_columns.append(col_name)
        return related_columns
    
    def _perform_cross_column_analysis(self, table_name: str, column_name: str, schema: Dict) -> Dict:
        """Perform cross-column analysis to understand data relationships"""
        analysis = {
            "joins_suggested": [],
            "grouping_potential": [],
            "filtering_potential": []
        }
        
        column_info = schema.get(column_name, {})
        purpose = column_info.get("inferred_purpose", {})
        category = purpose.get("category", "unknown")
        
        # Suggest joins based on identifiers
        if category == "identifier":
            analysis["joins_suggested"].append({
                "join_type": "potential_parent_child",
                "reasoning": f"{column_name} appears to be an identifier suitable for joins"
            })
        
        # Suggest grouping based on categorical columns
        if category in ["categorical", "status", "geographic"]:
            analysis["grouping_potential"].append({
                "group_by_usage": f"GROUP BY {column_name}",
                "business_use": "Segment analysis and categorical reporting"
            })
        
        # Suggest filtering based on low cardinality
        stats = column_info.get("sample_statistics", {})
        cardinality_ratio = stats.get("cardinality_ratio", 1)
        
        if cardinality_ratio < 0.1:
            analysis["filtering_potential"].append({
                "filter_usage": f"WHERE {column_name} = 'value'",
                "business_use": "Filter data for specific categories or status values"
            })
        
        return analysis
    
    def _generate_correlation_summary(self, correlations: Dict, relationships: Dict, workflows: Dict) -> Dict:
        """Generate summary of correlation analysis"""
        return {
            "significant_correlations_found": len(correlations),
            "strongest_correlation": max(correlations.items(), key=lambda x: abs(x[1]["correlation"])) if correlations else None,
            "foreign_key_relationships": relationships.get("detected", False),
            "business_workflows_identified": len(workflows),
            "analysis_confidence": "high" if correlations or workflows else "low"
        }


class WeaviateKnowledgeSearchTool(BaseTool):
    name: str = "Corporate Knowledge Search"
    description: str = "Search existing Weaviate metadata to find similar business patterns and ensure consistency"
    
    def _run(self, search_context: str, search_type: str = "semantic") -> str:
        """Search existing metadata for patterns and consistency"""
        try:
            weaviate_client = WeaviateClientSingleton.get_instance()
            
            if not weaviate_client:
                return json.dumps({
                    "warning": "Weaviate not available",
                    "analysis": "Cannot search existing patterns - will work with new data only"
                })
            
            if search_type == "semantic":
                results = self._semantic_similarity_search(weaviate_client, search_context)
            elif search_type == "business_context":
                results = self._business_context_search(weaviate_client, search_context)
            else:
                results = self._general_pattern_search(weaviate_client, search_context)
            
            return json.dumps({
                "search_context": search_context,
                "search_type": search_type,
                "results": results,
                "patterns_found": len(results.get("similar_items", [])),
                "recommendations": self._generate_consistency_recommendations(results)
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Weaviate knowledge search failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _semantic_similarity_search(self, client, context: str) -> Dict:
        """Search for semantically similar datasets and columns"""
        try:
            # Search DatasetMetadata
            dataset_collection = client.collections.get("DatasetMetadata")
            dataset_results = dataset_collection.query.near_text(
                query=context,
                limit=3,
                return_properties=["tableName", "description", "businessPurpose", "tags"],
                return_metadata=wq.MetadataQuery(distance=True)
            )
            
            # Search Column metadata
            column_collection = client.collections.get("Column")
            column_results = column_collection.query.near_text(
                query=context,
                limit=5,
                return_properties=["columnName", "semanticType", "businessName", "description"],
                return_metadata=wq.MetadataQuery(distance=True)
            )
            
            return {
                "similar_datasets": [
                    {
                        "table_name": obj.properties.get("tableName"),
                        "description": obj.properties.get("description"),
                        "business_purpose": obj.properties.get("businessPurpose"),
                        "similarity": 1 - obj.metadata.distance
                    }
                    for obj in dataset_results.objects
                ],
                "similar_columns": [
                    {
                        "column_name": obj.properties.get("columnName"),
                        "semantic_type": obj.properties.get("semanticType"),
                        "business_name": obj.properties.get("businessName"),
                        "description": obj.properties.get("description"),
                        "similarity": 1 - obj.metadata.distance
                    }
                    for obj in column_results.objects
                ]
            }
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return {"error": str(e)}
    
    def _business_context_search(self, client, context: str) -> Dict:
        """Search for business context patterns"""
        # This would search for specific business contexts
        # For now, return semantic search results
        return self._semantic_similarity_search(client, context)
    
    def _general_pattern_search(self, client, context: str) -> Dict:
        """General pattern search"""
        return self._semantic_similarity_search(client, context)
    
    def _generate_consistency_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations for consistency"""
        recommendations = []
        
        similar_datasets = results.get("similar_datasets", [])
        similar_columns = results.get("similar_columns", [])
        
        if similar_datasets:
            recommendations.append(
                f"Found {len(similar_datasets)} similar datasets - ensure consistent business terminology"
            )
        
        if similar_columns:
            semantic_types = set(col.get("semantic_type") for col in similar_columns if col.get("semantic_type"))
            if semantic_types:
                recommendations.append(
                    f"Consider using semantic types: {', '.join(semantic_types)} for consistency"
                )
        
        if not similar_datasets and not similar_columns:
            recommendations.append("No similar patterns found - this appears to be a new data pattern")
        
        return recommendations