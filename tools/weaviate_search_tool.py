"""
WeaviateSearchTool - Pattern discovery for moving services metadata consistency
"""

import json
import logging
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from tools.weaviate_client import WeaviateClientSingleton
import weaviate.classes.query as wq

logger = logging.getLogger(__name__)

class WeaviateSearchTool(BaseTool):
    name: str = "Weaviate Pattern Search"
    description: str = "Search existing moving services metadata for similar patterns to ensure consistency and leverage established naming conventions"
    
    def _run(self, search_context: str, search_type: str = "pattern_discovery") -> str:
        """
        Search for similar patterns in existing Weaviate metadata
        
        Args:
            search_context: Business context to search for (e.g., "crew size operational metric")
            search_type: Type of search - "pattern_discovery", "column_similarity", or "dataset_context"
        """
        try:
            weaviate_client = WeaviateClientSingleton.get_instance()
            
            if not weaviate_client:
                return json.dumps({
                    "status": "cold_start",
                    "message": "Weaviate not available - no existing patterns to reference",
                    "similar_patterns": [],
                    "consistency_guidance": "Generate fresh metadata without existing pattern constraints"
                })
            
            # Test if collections exist and have data
            collections_status = self._check_collections_status(weaviate_client)
            
            if not collections_status["has_data"]:
                return json.dumps({
                    "status": "cold_start",
                    "message": "No existing metadata found - this appears to be the first table",
                    "collections_checked": collections_status["collections"],
                    "similar_patterns": [],
                    "consistency_guidance": "Generate initial metadata patterns that future tables can reference"
                })
            
            # Perform the search based on type
            if search_type == "column_similarity":
                results = self._search_similar_columns(weaviate_client, search_context)
            elif search_type == "dataset_context":
                results = self._search_similar_datasets(weaviate_client, search_context)
            else:  # pattern_discovery (default)
                results = self._comprehensive_pattern_search(weaviate_client, search_context)
            
            return json.dumps({
                "status": "patterns_found",
                "search_context": search_context,
                "search_type": search_type,
                "similar_patterns": results,
                "consistency_guidance": self._generate_consistency_guidance(results),
                "pattern_count": len(results.get("similar_items", []))
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            return json.dumps({
                "status": "search_error",
                "error": str(e),
                "fallback_guidance": "Proceed with metadata generation using moving services industry best practices"
            })
    
    def _check_collections_status(self, client) -> Dict[str, Any]:
        """Check if Weaviate collections exist and contain data"""
        try:
            collections = client.collections.list_all()
            collection_names = [col.name for col in collections] if collections else []
            
            status = {
                "collections": collection_names,
                "has_data": False,
                "data_counts": {}
            }
            
            # Check for required collections and their data
            required_collections = ["DatasetMetadata", "Column"]
            
            for collection_name in required_collections:
                if collection_name in collection_names:
                    try:
                        collection = client.collections.get(collection_name)
                        # Quick count check
                        sample_query = collection.query.fetch_objects(limit=1)
                        count = len(sample_query.objects)
                        status["data_counts"][collection_name] = count
                        
                        if count > 0:
                            status["has_data"] = True
                            
                    except Exception as e:
                        logger.warning(f"Could not check {collection_name}: {e}")
                        status["data_counts"][collection_name] = "unknown"
            
            return status
            
        except Exception as e:
            logger.warning(f"Collections status check failed: {e}")
            return {"collections": [], "has_data": False, "data_counts": {}}
    
    def _search_similar_columns(self, client, search_context: str) -> Dict[str, Any]:
        """Search for similar columns using semantic search across vectorized fields"""
        try:
            column_collection = client.collections.get("Column")
            
            # Search across all vectorized fields: description, businessName, semanticType
            results = column_collection.query.near_text(
                query=search_context,
                limit=8,
                return_properties=[
                    "columnName", "athenaDataType", "description", "businessName", 
                    "semanticType", "sampleValues", "sqlUsagePattern"
                ],
                return_metadata=wq.MetadataQuery(distance=True)
            )
            
            similar_columns = []
            for obj in results.objects:
                props = obj.properties
                similarity = 1 - obj.metadata.distance
                
                # Only include reasonably similar results
                if similarity > 0.6:
                    similar_columns.append({
                        "column_name": props.get("columnName", ""),
                        "business_name": props.get("businessName", ""),
                        "description": props.get("description", ""),
                        "semantic_type": props.get("semanticType", ""),
                        "data_type": props.get("athenaDataType", ""),
                        "sample_values": props.get("sampleValues", []),
                        "sql_pattern": props.get("sqlUsagePattern", ""),
                        "similarity_score": round(similarity, 3),
                        "pattern_type": "column_similarity"
                    })
            
            return {
                "search_type": "column_similarity",
                "similar_items": similar_columns,
                "search_query": search_context
            }
            
        except Exception as e:
            logger.warning(f"Column similarity search failed: {e}")
            return {"search_type": "column_similarity", "similar_items": [], "error": str(e)}
    
    def _search_similar_datasets(self, client, search_context: str) -> Dict[str, Any]:
        """Search for similar datasets using semantic search on description and answerableQuestions"""
        try:
            dataset_collection = client.collections.get("DatasetMetadata")
            
            # Search across vectorized fields: description, answerableQuestions
            results = dataset_collection.query.near_text(
                query=search_context,
                limit=5,
                return_properties=[
                    "tableName", "description", "answerableQuestions", 
                    "dataOwner", "sourceSystem"
                ],
                return_metadata=wq.MetadataQuery(distance=True)
            )
            
            similar_datasets = []
            for obj in results.objects:
                props = obj.properties
                similarity = 1 - obj.metadata.distance
                
                if similarity > 0.5:
                    # Parse answerable questions if they're JSON
                    answerable_questions = []
                    try:
                        questions_json = props.get("answerableQuestions", "[]")
                        answerable_questions = json.loads(questions_json) if isinstance(questions_json, str) else questions_json
                    except:
                        pass
                    
                    similar_datasets.append({
                        "table_name": props.get("tableName", ""),
                        "description": props.get("description", ""),
                        "answerable_questions": answerable_questions,
                        "data_owner": props.get("dataOwner", ""),
                        "source_system": props.get("sourceSystem", ""),
                        "similarity_score": round(similarity, 3),
                        "pattern_type": "dataset_context"
                    })
            
            return {
                "search_type": "dataset_context", 
                "similar_items": similar_datasets,
                "search_query": search_context
            }
            
        except Exception as e:
            logger.warning(f"Dataset similarity search failed: {e}")
            return {"search_type": "dataset_context", "similar_items": [], "error": str(e)}
    
    def _comprehensive_pattern_search(self, client, search_context: str) -> Dict[str, Any]:
        """Comprehensive search across both columns and datasets"""
        try:
            column_results = self._search_similar_columns(client, search_context)
            dataset_results = self._search_similar_datasets(client, search_context)
            
            # Combine and organize results
            comprehensive_results = {
                "search_type": "comprehensive_pattern_discovery",
                "search_query": search_context,
                "column_patterns": column_results.get("similar_items", []),
                "dataset_patterns": dataset_results.get("similar_items", []),
                "total_patterns_found": len(column_results.get("similar_items", [])) + len(dataset_results.get("similar_items", []))
            }
            
            # Extract key patterns for consistency guidance
            comprehensive_results["pattern_insights"] = self._extract_pattern_insights(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            logger.warning(f"Comprehensive pattern search failed: {e}")
            return {
                "search_type": "comprehensive_pattern_discovery", 
                "column_patterns": [], 
                "dataset_patterns": [],
                "error": str(e)
            }
    
    def _extract_pattern_insights(self, results: Dict) -> Dict[str, Any]:
        """Extract key insights from search results for consistency guidance"""
        insights = {
            "common_business_names": [],
            "semantic_type_patterns": [],
            "description_patterns": [],
            "naming_conventions": []
        }
        
        # Analyze column patterns
        for col in results.get("column_patterns", []):
            business_name = col.get("business_name", "")
            semantic_type = col.get("semantic_type", "")
            
            if business_name and business_name not in insights["common_business_names"]:
                insights["common_business_names"].append(business_name)
            
            if semantic_type and semantic_type not in insights["semantic_type_patterns"]:
                insights["semantic_type_patterns"].append(semantic_type)
        
        # Extract naming convention patterns
        business_names = [col.get("business_name", "") for col in results.get("column_patterns", [])]
        for name in business_names:
            if name:
                # Look for common patterns: "Crew Size", "Hourly Rate", etc.
                if " " in name and name.title() == name:
                    insights["naming_conventions"].append("title_case_with_spaces")
                    break
        
        return insights
    
    def _generate_consistency_guidance(self, search_results: Dict) -> List[str]:
        """Generate guidance for maintaining consistency with existing patterns"""
        guidance = []
        
        if "similar_items" in search_results:
            items = search_results["similar_items"]
        elif "column_patterns" in search_results:
            items = search_results["column_patterns"] + search_results.get("dataset_patterns", [])
        else:
            items = []
        
        if not items:
            guidance.append("No existing patterns found - establish new metadata standards for this data type")
            return guidance
        
        # Business naming guidance
        business_names = [item.get("business_name", "") for item in items if item.get("business_name")]
        if business_names:
            guidance.append(f"Use similar business naming patterns: {', '.join(business_names[:3])}")
        
        # Semantic type guidance
        semantic_types = list(set([item.get("semantic_type", "") for item in items if item.get("semantic_type")]))
        if semantic_types:
            guidance.append(f"Consider these semantic types for similar data: {', '.join(semantic_types)}")
        
        # Description style guidance
        descriptions = [item.get("description", "") for item in items if item.get("description")]
        if descriptions:
            avg_length = sum(len(desc.split()) for desc in descriptions) / len(descriptions)
            guidance.append(f"Maintain description style consistent with existing metadata (avg {int(avg_length)} words)")
        
        # Pattern-specific guidance
        if len(items) >= 3:
            guidance.append("Strong existing patterns found - prioritize consistency with established metadata")
        else:
            guidance.append("Limited existing patterns - balance consistency with optimal metadata for this specific data")
        
        return guidance