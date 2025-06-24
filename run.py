"""
Main runner for CrewAI Moving Services Metadata Generation Pipeline
"""

import os
import sys
import json
import yaml
import logging
import boto3
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from crewai import Crew, Process
from tasks import create_metadata_tasks
from tools.weaviate_client import WeaviateClientSingleton
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovingServicesMetadataGenerator:
    """Main orchestrator for moving services metadata generation"""
    
    def __init__(self):
        self.athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.glue_client = boto3.client('glue', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        
        # Configuration
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate environment
        self._validate_environment()
    
    def _validate_environment(self):
        """Validate environment setup"""
        required_env_vars = [
            'AWS_REGION',
            'ATHENA_RESULTS_BUCKET',
            'WEAVIATE_URL'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        logger.info("✅ Environment validation passed")
    
    def get_glue_table_metadata(self, database_name: str, table_name: str) -> Dict[str, Any]:
        """Get table metadata from AWS Glue"""
        try:
            logger.info(f"📊 Fetching Glue metadata for {database_name}.{table_name}")
            
            response = self.glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            
            table_info = response['Table']
            
            # Extract relevant metadata
            metadata = {
                "table_name": table_info['Name'],
                "database_name": database_name,
                "full_table_name": f"{database_name}.{table_name}",
                "description": table_info.get('Description', ''),
                "location": table_info.get('StorageDescriptor', {}).get('Location', ''),
                "input_format": table_info.get('StorageDescriptor', {}).get('InputFormat', ''),
                "output_format": table_info.get('StorageDescriptor', {}).get('OutputFormat', ''),
                "created_time": table_info.get('CreateTime', '').isoformat() if table_info.get('CreateTime') else '',
                "updated_time": table_info.get('UpdateTime', '').isoformat() if table_info.get('UpdateTime') else '',
                "columns": []
            }
            
            # Extract column information
            columns = table_info.get('StorageDescriptor', {}).get('Columns', [])
            for col in columns:
                column_info = {
                    "name": col.get('Name', ''),
                    "type": col.get('Type', ''),
                    "comment": col.get('Comment', '')
                }
                metadata["columns"].append(column_info)
            
            logger.info(f"✅ Retrieved metadata for {len(columns)} columns")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to get Glue metadata: {e}")
            raise
    
    def test_athena_connectivity(self, table_name: str) -> bool:
        """Test Athena connectivity with the table"""
        try:
            test_query = f"SELECT COUNT(*) as record_count FROM {table_name} LIMIT 1"
            
            response = self.athena_client.start_query_execution(
                QueryString=test_query,
                ResultConfiguration={
                    'OutputLocation': os.getenv('ATHENA_RESULTS_BUCKET')
                },
                WorkGroup=os.getenv('ATHENA_WORKGROUP', 'primary')
            )
            
            query_id = response['QueryExecutionId']
            
            # Wait for query completion (simple polling)
            import time
            for _ in range(10):  # Wait up to 20 seconds
                result = self.athena_client.get_query_execution(QueryExecutionId=query_id)
                status = result['QueryExecution']['Status']['State']
                
                if status == 'SUCCEEDED':
                    logger.info("✅ Athena connectivity test passed")
                    return True
                elif status in ['FAILED', 'CANCELLED']:
                    logger.warning(f"⚠️ Athena test query failed: {status}")
                    return False
                
                time.sleep(2)
            
            logger.warning("⚠️ Athena test query timed out")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Athena connectivity test failed: {e}")
            return False
    
    def run_metadata_generation(self, database_name: str, table_name: str) -> Dict[str, Any]:
        """Run the complete metadata generation pipeline"""
        
        logger.info(f"🚀 Starting metadata generation for {database_name}.{table_name}")
        
        # Step 1: Get Glue metadata
        try:
            glue_metadata = self.get_glue_table_metadata(database_name, table_name)
        except Exception as e:
            logger.error(f"Failed to get Glue metadata: {e}")
            return {"error": f"Glue metadata retrieval failed: {str(e)}"}
        
        # Step 2: Test Athena connectivity
        full_table_name = f"{database_name}.{table_name}"
        athena_available = self.test_athena_connectivity(full_table_name)
        
        if not athena_available:
            logger.warning("⚠️ Athena not available - proceeding with limited analysis")
        
        # Step 3: Test Weaviate connectivity
        weaviate_client = WeaviateClientSingleton.get_instance()
        weaviate_available = WeaviateClientSingleton.test_connection()
        
        if not weaviate_available:
            logger.warning("⚠️ Weaviate not available - proceeding without knowledge search")
        
        # Step 4: Create and run CrewAI crew
        try:
            logger.info("🤖 Creating CrewAI crew for metadata generation")
            
            # Create tasks with dynamic inputs
            pattern_task, synthesis_task, validation_task = create_metadata_tasks(
                table_name=full_table_name,
                glue_metadata=glue_metadata
            )
            
            # Create crew
            metadata_crew = Crew(
                agents=[
                    pattern_task.agent,
                    synthesis_task.agent,
                    validation_task.agent
                ],
                tasks=[pattern_task, synthesis_task, validation_task],
                process=Process.sequential,
                verbose=True,
                memory=False  # Disable memory to avoid OpenAI embeddings
            )
            
            logger.info("🚀 Starting crew execution...")
            
            # Execute crew
            result = metadata_crew.kickoff(inputs={
                'table_name': full_table_name,
                'glue_metadata': glue_metadata,
                'athena_available': athena_available,
                'weaviate_available': weaviate_available
            })
            
            logger.info("✅ Crew execution completed")
            
            # Step 5: Process results
            return self._process_crew_results(result, database_name, table_name)
            
        except Exception as e:
            logger.error(f"❌ Crew execution failed: {e}")
            return {"error": f"Crew execution failed: {str(e)}"}
    
    def _process_crew_results(self, crew_result, database_name: str, table_name: str) -> Dict[str, Any]:
        """Process and save crew results"""
        
        logger.info("📋 Processing crew results...")
        
        try:
            # Get the raw output from crew execution
            if hasattr(crew_result, 'raw'):
                final_output = crew_result.raw
            else:
                final_output = str(crew_result)
            
            # Get task outputs to extract YAML and validation separately
            task_outputs = []
            if hasattr(crew_result, 'tasks_output'):
                task_outputs = crew_result.tasks_output
            
            # Initialize variables for YAML and validation
            yaml_content = None
            validation_result = None
            
            # Process task outputs to find YAML and validation
            for i, task_output in enumerate(task_outputs):
                output_content = task_output.raw if hasattr(task_output, 'raw') else str(task_output)
                
                # Task 1: Pattern Discovery (JSON analysis)
                if i == 0:
                    logger.info("📊 Pattern discovery analysis completed")
                
                # Task 2: Business Synthesis (YAML output)
                elif i == 1:
                    if output_content.strip().startswith('```yaml') or 'DatasetMetadata:' in output_content:
                        yaml_content = self._extract_yaml_from_output(output_content)
                        logger.info("📝 YAML metadata extracted")
                
                # Task 3: Validation (JSON validation result)
                elif i == 2:
                    try:
                        validation_result = json.loads(output_content)
                        logger.info("✅ Validation result extracted")
                    except json.JSONDecodeError:
                        # If validation output is not JSON, create a simple result
                        validation_result = {
                            "yaml_validation_status": "COMPLETED",
                            "raw_validation_output": output_content
                        }
            
            # Fallback: try to parse final output
            if not validation_result:
                try:
                    validation_result = json.loads(final_output)
                except json.JSONDecodeError:
                    validation_result = {"raw_output": final_output}
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save YAML file if extracted
            yaml_file = None
            if yaml_content:
                yaml_file = self.output_dir / f"{database_name}_{table_name}_{timestamp}.yaml"
                with open(yaml_file, 'w') as f:
                    f.write(yaml_content)
                logger.info(f"📄 YAML metadata saved to {yaml_file}")
            
            # Save validation results as JSON
            json_file = self.output_dir / f"{database_name}_{table_name}_{timestamp}_validation.json"
            with open(json_file, 'w') as f:
                json.dump({
                    "database_name": database_name,
                    "table_name": table_name,
                    "generation_timestamp": timestamp,
                    "validation_result": validation_result,
                    "yaml_file_generated": str(yaml_file) if yaml_file else None,
                    "crew_execution_status": "completed"
                }, f, indent=2)
            
            logger.info(f"💾 Validation results saved to {json_file}")
            
            # Check validation status
            status = validation_result.get("yaml_validation_status", "UNKNOWN")
            
            if status == "APPROVED":
                logger.info("✅ Metadata APPROVED - ready for Weaviate ingestion")
                
            elif status == "REJECTED":
                logger.warning("❌ Metadata REJECTED - improvements needed")
                improvements = validation_result.get("improvement_recommendations", [])
                for improvement in improvements:
                    logger.warning(f"   💡 {improvement}")
                    
            else:
                logger.info(f"ℹ️ Metadata validation status: {status}")
            
            return {
                "status": "success",
                "validation_result": validation_result,
                "yaml_file": str(yaml_file) if yaml_file else None,
                "validation_file": str(json_file),
                "approved_for_ingestion": status == "APPROVED"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process crew results: {e}")
            return {"error": f"Failed to process results: {str(e)}"}
    
    def _extract_yaml_from_output(self, output_content: str) -> str:
        """Extract YAML content from agent output"""
        try:
            # Remove markdown code blocks if present
            if '```yaml' in output_content:
                start = output_content.find('```yaml') + 7
                end = output_content.find('```', start)
                if end != -1:
                    return output_content[start:end].strip()
            
            # If no code blocks, look for YAML content starting with DatasetMetadata
            if 'DatasetMetadata:' in output_content:
                start = output_content.find('DatasetMetadata:')
                return output_content[start:].strip()
            
            # Return the whole content as fallback
            return output_content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to extract YAML: {e}")
            return output_content


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate moving services metadata using CrewAI")
    parser.add_argument("--database", required=True, help="Glue database name")
    parser.add_argument("--table", required=True, help="Glue table name")
    parser.add_argument("--test-connections", action="store_true", help="Test connections only")
    
    args = parser.parse_args()
    
    try:
        generator = MovingServicesMetadataGenerator()
        
        if args.test_connections:
            logger.info("🧪 Testing connections...")
            
            # Test Weaviate
            weaviate_ok = WeaviateClientSingleton.test_connection()
            logger.info(f"Weaviate: {'✅ OK' if weaviate_ok else '❌ Failed'}")
            
            # Test Athena
            athena_ok = generator.test_athena_connectivity(f"{args.database}.{args.table}")
            logger.info(f"Athena: {'✅ OK' if athena_ok else '❌ Failed'}")
            
            return
        
        # Run metadata generation
        result = generator.run_metadata_generation(args.database, args.table)
        
        if "error" in result:
            logger.error(f"❌ Generation failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info("🎉 Metadata generation completed successfully!")
            if result.get("approved_for_ingestion"):
                logger.info("✅ Metadata approved and ready for Weaviate ingestion")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up connections
        WeaviateClientSingleton.close()


if __name__ == "__main__":
    main()