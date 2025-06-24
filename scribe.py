"""
Local Scribe implementation - converts Glue metadata to draft YAML
Based on your existing Lambda function but adapted for local execution
"""

import boto3
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalScribe:
    """Local implementation of the Scribe for draft YAML generation"""
    
    def __init__(self, database_name: str = None):
        self.glue = boto3.client('glue')
        self.database_name = database_name or 'ams-dataset-metaset-extractor-test'
        
        # Local storage instead of S3
        self.drafts_dir = Path("drafts")
        self.drafts_dir.mkdir(exist_ok=True)
        
        logger.info(f"🖋️ Scribe initialized for database: {self.database_name}")
    
    def get_latest_table(self) -> Dict[str, str]:
        """Get the most recently updated table from Glue database"""
        try:
            logger.info(f"📊 Fetching tables from {self.database_name}")
            
            tables = self.glue.get_tables(DatabaseName=self.database_name)['TableList']
            if not tables:
                raise ValueError(f"No tables found in database {self.database_name}")
            
            # Sort by update time (most recent first)
            tables.sort(
                key=lambda x: x.get('UpdateTime') or x.get('CreateTime'), 
                reverse=True
            )
            
            latest_table = tables[0]['Name']
            logger.info(f"📅 Latest table: {latest_table}")
            
            return {
                "tableName": latest_table,
                "databaseName": self.database_name
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest table: {e}")
            raise
    
    def create_draft_yaml(self, table_name: str, database_name: str = None) -> Dict[str, Any]:
        """Create draft YAML from Glue table metadata - matches your Lambda logic"""
        
        db_name = database_name or self.database_name
        
        try:
            logger.info(f"🖋️ Creating draft YAML for {db_name}.{table_name}")
            
            # Fetch full table details from Glue
            table = self.glue.get_table(DatabaseName=db_name, Name=table_name)['Table']
            
            # Extract table-level comment
            table_comment = table.get("Parameters", {}).get("comment", "")
            
            # Build YAML structure - matching your Lambda format
            yaml_content = {
                "datasetMetadata": {
                    "tableName": table_name,
                    "athenaTableName": f"{db_name}.{table_name}",  # Full Athena table name
                    "description": table_comment  # Will be enhanced by AI
                },
                "columns": []
            }
            
            # Process columns
            columns = table.get('StorageDescriptor', {}).get('Columns', [])
            logger.info(f"📋 Processing {len(columns)} columns")
            
            for column in columns:
                column_entry = {
                    "columnName": column['Name'],
                    "athenaDataType": column['Type'],
                    "parentAthenaTableName": f"{db_name}.{table_name}",
                    "description": column.get("Comment", "")  # Will be enhanced by AI
                }
                yaml_content["columns"].append(column_entry)
            
            # Save draft YAML locally (instead of S3)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{db_name}_{table_name}_{timestamp}.yaml"
            filepath = self.drafts_dir / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
            
            logger.info(f"✅ Draft YAML saved: {filepath}")
            
            return {
                "yaml_content": yaml_content,
                "draft_file": str(filepath),
                "table_metadata": {
                    "table_name": table_name,
                    "database_name": db_name,
                    "full_table_name": f"{db_name}.{table_name}",
                    "columns_count": len(columns),
                    "has_table_comment": bool(table_comment)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create draft YAML: {e}")
            raise
    
    def process_table(self, table_name: str = None, database_name: str = None) -> Dict[str, Any]:
        """Process a specific table or get the latest one"""
        
        if not table_name:
            # Get latest table
            latest_info = self.get_latest_table()
            table_name = latest_info["tableName"]
            database_name = latest_info["databaseName"]
        
        return self.create_draft_yaml(table_name, database_name)
    
    def list_available_tables(self) -> List[str]:
        """List all available tables in the database"""
        try:
            tables = self.glue.get_tables(DatabaseName=self.database_name)['TableList']
            table_names = [table['Name'] for table in tables]
            logger.info(f"📊 Available tables: {table_names}")
            return table_names
        except Exception as e:
            logger.error(f"❌ Failed to list tables: {e}")
            return []