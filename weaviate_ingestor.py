"""
Weaviate ingestion component for AI-generated metadata
"""

import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from tools.weaviate_client import WeaviateClientSingleton

logger = logging.getLogger(__name__)

class WeaviateIngestor:
    """Handles ingestion of AI-generated metadata into Weaviate"""
    
    def __init__(self):
        self.client = WeaviateClientSingleton.get_instance()
        if not self.client:
            logger.warning("⚠️ Weaviate client not available - ingestion will be simulated")
    
    def ingest_yaml_metadata(self, yaml_file_path: str) -> Dict[str, Any]:
        """Ingest YAML metadata into Weaviate collections"""
        
        try:
            logger.info(f"📥 Starting Weaviate ingestion from {yaml_file_path}")
            
            # Load and parse YAML
            with open(yaml_file_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            if not metadata:
                raise ValueError("Empty YAML file")
            
            # Validate YAML structure
            if "DatasetMetadata" not in metadata or "Column" not in metadata:
                raise ValueError("YAML missing required sections: DatasetMetadata, Column")
            
            ingestion_results = {
                "dataset_ingestion": self._ingest_dataset_metadata(metadata["DatasetMetadata"]),
                "column_ingestion": self._ingest_column_metadata(metadata["Column"]),
                "total_objects_created": 0,
                "ingestion_status": "success"
            }
            
            # Calculate totals
            dataset_created = 1 if ingestion_results["dataset_ingestion"]["status"] == "success" else 0
            columns_created = ingestion_results["column_ingestion"]["objects_created"]
            ingestion_results["total_objects_created"] = dataset_created + columns_created
            
            logger.info(f"✅ Ingestion completed: {ingestion_results['total_objects_created']} objects created")
            
            return ingestion_results
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            return {
                "ingestion_status": "failed",
                "error": str(e),
                "total_objects_created": 0
            }
    
    def _ingest_dataset_metadata(self, dataset_data: Dict) -> Dict[str, Any]:
        """Ingest DatasetMetadata object"""
        
        try:
            if not self.client:
                logger.info("📝 Simulating DatasetMetadata ingestion (no Weaviate client)")
                return {
                    "status": "simulated",
                    "table_name": dataset_data.get("tableName", "unknown"),
                    "object_id": "simulated-uuid-dataset"
                }
            
            # Get DatasetMetadata collection
            dataset_collection = self.client.collections.get("DatasetMetadata")
            
            # Prepare data for Weaviate
            weaviate_data = {
                "tableName": dataset_data.get("tableName", ""),
                "athenaTableName": dataset_data.get("athenaTableName", ""),
                "description": dataset_data.get("description", ""),
                "answerableQuestions": dataset_data.get("answerableQuestions", "[]"),
                "llmHints": dataset_data.get("llmHints", "{}"),
                "dataOwner": dataset_data.get("dataOwner", "Engineering Team"),
                "sourceSystem": dataset_data.get("sourceSystem", "AWS Glue Catalog")
            }
            
            # Validate JSON fields
            try:
                json.loads(weaviate_data["answerableQuestions"])
                json.loads(weaviate_data["llmHints"])
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON validation warning: {e}")
            
            # Insert into Weaviate
            object_id = dataset_collection.data.insert(weaviate_data)
            
            logger.info(f"✅ DatasetMetadata created: {weaviate_data['tableName']} (ID: {object_id})")
            
            return {
                "status": "success",
                "table_name": weaviate_data["tableName"],
                "object_id": str(object_id)
            }
            
        except Exception as e:
            logger.error(f"❌ DatasetMetadata ingestion failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "table_name": dataset_data.get("tableName", "unknown")
            }
    
    def _ingest_column_metadata(self, column_data: List[Dict]) -> Dict[str, Any]:
        """Ingest Column objects"""
        
        results = {
            "status": "success",
            "objects_created": 0,
            "failed_objects": 0,
            "column_details": []
        }
        
        try:
            if not self.client:
                logger.info(f"📝 Simulating {len(column_data)} Column ingestions (no Weaviate client)")
                return {
                    "status": "simulated",
                    "objects_created": len(column_data),
                    "failed_objects": 0,
                    "column_details": [{"column": col.get("columnName", "unknown"), "status": "simulated"} for col in column_data]
                }
            
            # Get Column collection
            column_collection = self.client.collections.get("Column")
            
            for column in column_data:
                try:
                    # Prepare column data for Weaviate
                    weaviate_column_data = {
                        "columnName": column.get("columnName", ""),
                        "athenaDataType": column.get("athenaDataType", "string"),
                        "parentAthenaTableName": column.get("parentAthenaTableName", ""),
                        "description": column.get("description", ""),
                        "businessName": column.get("businessName", ""),
                        "semanticType": column.get("semanticType", "general"),
                        "sqlUsagePattern": column.get("sqlUsagePattern", ""),
                        "sampleValues": column.get("sampleValues", []),
                        "foreignKeyInfo": column.get("foreignKeyInfo", "{}"),
                        "isPrimaryKey": column.get("isPrimaryKey", False),
                        "dataClassification": column.get("dataClassification", "Internal")
                    }
                    
                    # Validate foreignKeyInfo JSON
                    try:
                        json.loads(weaviate_column_data["foreignKeyInfo"])
                    except json.JSONDecodeError:
                        weaviate_column_data["foreignKeyInfo"] = "{}"
                    
                    # Insert into Weaviate
                    object_id = column_collection.data.insert(weaviate_column_data)
                    
                    results["objects_created"] += 1
                    results["column_details"].append({
                        "column": weaviate_column_data["columnName"],
                        "status": "success",
                        "object_id": str(object_id)
                    })
                    
                    logger.info(f"✅ Column created: {weaviate_column_data['columnName']} (ID: {object_id})")
                    
                except Exception as e:
                    results["failed_objects"] += 1
                    results["column_details"].append({
                        "column": column.get("columnName", "unknown"),
                        "status": "failed",
                        "error": str(e)
                    })
                    logger.error(f"❌ Column ingestion failed for {column.get('columnName', 'unknown')}: {e}")
            
            if results["failed_objects"] > 0:
                results["status"] = "partial_success"
            
            logger.info(f"📊 Column ingestion summary: {results['objects_created']} created, {results['failed_objects']} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Column ingestion process failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "objects_created": 0,
                "failed_objects": len(column_data)
            }
    
    def verify_ingestion(self, table_name: str) -> Dict[str, Any]:
        """Verify that ingested data is available in Weaviate"""
        
        if not self.client:
            return {"status": "skipped", "reason": "No Weaviate client available"}
        
        try:
            verification_results = {}
            
            # Check DatasetMetadata
            dataset_collection = self.client.collections.get("DatasetMetadata")
            dataset_query = dataset_collection.query.fetch_objects(
                where={"path": ["tableName"], "operator": "Equal", "valueText": table_name},
                limit=1
            )
            
            verification_results["dataset_found"] = len(dataset_query.objects) > 0
            
            # Check Columns
            column_collection = self.client.collections.get("Column")
            column_query = column_collection.query.fetch_objects(
                where={"path": ["parentAthenaTableName"], "operator": "Like", "valueText": f"*{table_name}"},
                limit=10
            )
            
            verification_results["columns_found"] = len(column_query.objects)
            verification_results["verification_status"] = "success"
            
            logger.info(f"🔍 Verification: Dataset found: {verification_results['dataset_found']}, "
                       f"Columns found: {verification_results['columns_found']}")
            
            return verification_results
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return {
                "verification_status": "failed",
                "error": str(e)
            }
    
    def test_collections_available(self) -> Dict[str, bool]:
        """Test if required collections are available in Weaviate"""
        
        if not self.client:
            return {"DatasetMetadata": False, "Column": False, "client_available": False}
        
        try:
            collections = self.client.collections.list_all()
            collection_names = [col.name for col in collections] if collections else []
            
            available = {
                "DatasetMetadata": "DatasetMetadata" in collection_names,
                "Column": "Column" in collection_names,
                "client_available": True
            }
            
            logger.info(f"📋 Collections available: {available}")
            return available
            
        except Exception as e:
            logger.error(f"❌ Failed to check collections: {e}")
            return {"DatasetMetadata": False, "Column": False, "client_available": False}