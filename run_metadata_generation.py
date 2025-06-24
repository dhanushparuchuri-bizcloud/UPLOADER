# run_metadata_generation.py
"""
Complete metadata generation pipeline for moving services
Integrates with your existing scribe and saves YAML to output folder
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from crewai import Crew, Process, LLM
from dotenv import load_dotenv

# Import your existing scribe
from scribe import LocalScribe

# Import agents and tasks (you'll create these files)
from agents import (
    data_discovery_agent,
    business_intelligence_agent, 
    metadata_quality_validator
)
from tasks import (
    discover_data_patterns,
    generate_business_metadata,
    validate_metadata_quality
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovingServicesMetadataGenerator:
    """Complete metadata generation pipeline for moving services"""
    
    def __init__(self):
        self.scribe = LocalScribe(database_name=os.getenv('ATHENA_DATABASE', 'amspoc3test'))
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure Bedrock LLM for agents
        self.bedrock_llm = LLM(
            model=f"bedrock/{os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')}",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        logger.info("🚀 Moving Services Metadata Generator initialized")
    
    def run_complete_pipeline(self, table_name: str = None, database_name: str = None) -> Dict[str, Any]:
        """Run the complete metadata generation pipeline"""
        
        try:
            # Step 1: Get draft YAML from your existing scribe
            logger.info("📝 Step 1: Creating draft YAML from Glue metadata")
            draft_result = self.scribe.process_table(table_name, database_name)
            
            if not draft_result:
                raise ValueError("Failed to create draft YAML from Glue metadata")
            
            table_metadata = draft_result["table_metadata"]
            full_table_name = table_metadata["full_table_name"]
            logger.info(f"✅ Draft YAML created for {full_table_name}")
            
            # Step 2: Run CrewAI metadata enhancement
            logger.info("🤖 Step 2: Running CrewAI metadata enhancement")
            enhanced_result = self._run_crewai_enhancement(full_table_name, draft_result)
            
            if not enhanced_result.get("success"):
                raise ValueError(f"CrewAI enhancement failed: {enhanced_result.get('error')}")
            
            # Step 3: Save enhanced YAML to output folder
            logger.info("💾 Step 3: Saving enhanced YAML to output folder")
            output_files = self._save_enhanced_yaml(enhanced_result, table_metadata)
            
            # Step 4: Generate summary report
            summary = self._generate_summary_report(draft_result, enhanced_result, output_files)
            
            logger.info("🎉 Complete pipeline executed successfully!")
            return {
                "status": "success",
                "table_processed": full_table_name,
                "files_generated": output_files,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "table_name": table_name
            }
    
    def _run_crewai_enhancement(self, table_name: str, draft_result: Dict) -> Dict[str, Any]:
        """Run the CrewAI agents to enhance the metadata"""
        
        try:
            # Set LLM for all agents
            data_discovery_agent.llm = self.bedrock_llm
            business_intelligence_agent.llm = self.bedrock_llm  
            metadata_quality_validator.llm = self.bedrock_llm
            
            # Create the crew
            metadata_crew = Crew(
                agents=[
                    data_discovery_agent,
                    business_intelligence_agent,
                    metadata_quality_validator
                ],
                tasks=[
                    discover_data_patterns,
                    generate_business_metadata, 
                    validate_metadata_quality
                ],
                process=Process.sequential,
                verbose=True,
                memory=False  # Disable to avoid OpenAI embeddings
            )
            
            # Prepare inputs
            crew_inputs = {
                'table_name': table_name,
                'full_table_name': table_name,  # Add this for template compatibility
                'draft_yaml': draft_result["yaml_content"],
                'table_metadata': draft_result["table_metadata"]
            }
            
            logger.info(f"🤖 Starting CrewAI enhancement for {table_name}")
            
            # Execute the crew
            result = metadata_crew.kickoff(inputs=crew_inputs)
            
            # Process results
            final_yaml = self._extract_yaml_from_crew_result(result)
            validation_result = self._extract_validation_from_crew_result(result)
            
            return {
                "success": True,
                "final_yaml": final_yaml,
                "validation_result": validation_result,
                "crew_result": result
            }
            
        except Exception as e:
            logger.error(f"CrewAI enhancement failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_yaml_from_crew_result(self, crew_result) -> str:
        """Extract the final YAML from crew execution"""
        try:
            # Get the business intelligence agent's output (task 2)
            if hasattr(crew_result, 'tasks_output') and len(crew_result.tasks_output) >= 2:
                synthesis_output = crew_result.tasks_output[1]
                
                if hasattr(synthesis_output, 'raw'):
                    output_content = synthesis_output.raw
                else:
                    output_content = str(synthesis_output)
                
                # Extract YAML from markdown code blocks if present
                if '```yaml' in output_content:
                    start = output_content.find('```yaml') + 7
                    end = output_content.find('```', start)
                    if end != -1:
                        return output_content[start:end].strip()
                
                # Look for YAML content starting with DatasetMetadata
                if 'DatasetMetadata:' in output_content:
                    start = output_content.find('DatasetMetadata:')
                    return output_content[start:].strip()
            
            # Fallback: try to get from final result
            if hasattr(crew_result, 'raw'):
                content = crew_result.raw
                if 'DatasetMetadata:' in content:
                    start = content.find('DatasetMetadata:')
                    return content[start:].strip()
            
            raise ValueError("Could not extract YAML from crew result")
            
        except Exception as e:
            logger.warning(f"YAML extraction failed: {e}")
            return ""
    
    def _extract_validation_from_crew_result(self, crew_result) -> Dict:
        """Extract validation result from crew execution"""
        try:
            # Get the validation agent's output (task 3)
            if hasattr(crew_result, 'tasks_output') and len(crew_result.tasks_output) >= 3:
                validation_output = crew_result.tasks_output[2]
                
                if hasattr(validation_output, 'raw'):
                    output_content = validation_output.raw
                else:
                    output_content = str(validation_output)
                
                # Try to parse as JSON
                try:
                    return json.loads(output_content)
                except json.JSONDecodeError:
                    # Look for JSON in the content
                    json_match = re.search(r'\{.*\}', output_content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
            
            # Fallback validation result
            return {
                "validation_status": "COMPLETED",
                "message": "Validation result could not be parsed, but process completed"
            }
            
        except Exception as e:
            logger.warning(f"Validation extraction failed: {e}")
            return {"validation_status": "ERROR", "error": str(e)}
    
    def _save_enhanced_yaml(self, enhanced_result: Dict, table_metadata: Dict) -> Dict[str, str]:
        """Save the enhanced YAML to output folder"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            table_name = table_metadata["table_name"]
            database_name = table_metadata["database_name"]
            
            # Generate filenames
            yaml_filename = f"{database_name}_{table_name}_{timestamp}_enhanced.yaml"
            validation_filename = f"{database_name}_{table_name}_{timestamp}_validation.json"
            
            yaml_filepath = self.output_dir / yaml_filename
            validation_filepath = self.output_dir / validation_filename
            
            # Save enhanced YAML
            final_yaml = enhanced_result.get("final_yaml", "")
            if final_yaml:
                with open(yaml_filepath, 'w') as f:
                    f.write(final_yaml)
                logger.info(f"✅ Enhanced YAML saved: {yaml_filepath}")
            else:
                logger.warning("⚠️ No enhanced YAML content to save")
            
            # Save validation results
            validation_result = enhanced_result.get("validation_result", {})
            validation_data = {
                "table_name": table_name,
                "database_name": database_name,
                "generation_timestamp": timestamp,
                "validation_result": validation_result,
                "yaml_file": yaml_filename if final_yaml else None,
                "enhancement_status": "completed"
            }
            
            with open(validation_filepath, 'w') as f:
                json.dump(validation_data, f, indent=2)
            logger.info(f"✅ Validation results saved: {validation_filepath}")
            
            return {
                "yaml_file": str(yaml_filepath) if final_yaml else None,
                "validation_file": str(validation_filepath),
                "yaml_filename": yaml_filename if final_yaml else None,
                "validation_filename": validation_filename
            }
            
        except Exception as e:
            logger.error(f"Failed to save enhanced files: {e}")
            return {"error": str(e)}
    
    def _generate_summary_report(self, draft_result: Dict, enhanced_result: Dict, output_files: Dict) -> Dict:
        """Generate a summary report of the processing"""
        
        table_metadata = draft_result["table_metadata"]
        validation_result = enhanced_result.get("validation_result", {})
        
        summary = {
            "table_processed": {
                "name": table_metadata["table_name"],
                "full_name": table_metadata["full_table_name"],
                "columns_count": table_metadata["columns_count"]
            },
            "processing_status": {
                "draft_creation": "success",
                "ai_enhancement": "success" if enhanced_result.get("success") else "failed",
                "validation_status": validation_result.get("validation_status", "unknown"),
                "files_saved": len([f for f in output_files.values() if f and not f.startswith("error")])
            },
            "output_files": output_files,
            "metadata_quality": {
                "validation_score": validation_result.get("overall_quality_score", "N/A"),
                "business_intelligence": validation_result.get("business_intelligence_assessment", {}),
                "ready_for_weaviate": validation_result.get("validation_status") == "APPROVED"
            }
        }
        
        return summary


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced metadata for moving services tables")
    parser.add_argument("--table", help="Specific table name to process")
    parser.add_argument("--database", help="Database name (defaults to ATHENA_DATABASE env var)")
    parser.add_argument("--latest", action="store_true", help="Process the latest updated table")
    
    args = parser.parse_args()
    
    try:
        generator = MovingServicesMetadataGenerator()
        
        if args.latest:
            logger.info("🔍 Processing latest updated table")
            result = generator.run_complete_pipeline()
        else:
            table_name = args.table
            database_name = args.database
            
            if not table_name:
                logger.error("❌ Table name required. Use --table or --latest")
                sys.exit(1)
            
            logger.info(f"🔍 Processing table: {table_name}")
            result = generator.run_complete_pipeline(table_name, database_name)
        
        # Print summary
        if result["status"] == "success":
            summary = result["summary"]
            logger.info("📊 PROCESSING SUMMARY:")
            logger.info(f"   Table: {summary['table_processed']['full_name']}")
            logger.info(f"   Columns: {summary['table_processed']['columns_count']}")
            logger.info(f"   Validation: {summary['metadata_quality']['validation_score']}")
            logger.info(f"   Ready for Weaviate: {summary['metadata_quality']['ready_for_weaviate']}")
            
            if summary['output_files']['yaml_file']:
                logger.info(f"   YAML: {summary['output_files']['yaml_filename']}")
            
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error(f"❌ Pipeline failed: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()