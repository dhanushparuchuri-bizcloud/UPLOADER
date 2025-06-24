"""
CrewAI tasks for moving services metadata generation
"""

from crewai import Task
from agents import (
    pattern_discovery_profiler,
    business_intelligence_synthesizer,
    quality_assurance_validator
)

def create_pattern_discovery_task(table_name: str, glue_metadata: dict) -> Task:
    """Create pattern discovery task with dynamic inputs"""
    return Task(
        description=f"""Analyze the new table '{table_name}' with technical metadata from AWS Glue and discover 
        comprehensive business patterns using statistical analysis and moving services domain expertise.

        DISCOVERY MISSION:
        1. STATISTICAL CORRELATION ANALYSIS:
           - Find columns that frequently appear together in business workflows
           - Identify revenue impact relationships (crew size → labor costs → profitability)
           - Discover cost driver patterns (miles → fuel costs, hours → labor expenses)
           - Calculate statistical significance of relationships (correlation coefficients, sample sizes)

        2. SEASONAL PATTERN DETECTION:
           - Analyze temporal patterns for moving industry seasonality (summer peaks, winter slowdowns)
           - Detect school calendar correlations (college move-in/out periods)
           - Identify weather impact patterns on operations and costs
           - Find geographic seasonal variations (northern vs southern markets)

        3. GEOGRAPHIC VARIATION ANALYSIS:
           - Compare cost structures across different branch locations
           - Identify regional pricing differences and market maturity levels
           - Analyze demographic correlations (income levels, population density, homeownership rates)
           - Detect competitive landscape differences by geography

        4. MOVING SERVICES BUSINESS CONTEXT:
           - Labor efficiency indicators (crew productivity, completion times)
           - Equipment utilization patterns (truck usage, maintenance cycles)
           - Customer satisfaction drivers (service quality, delivery times)
           - Operational optimization opportunities (route efficiency, resource allocation)

        Create a comprehensive BUSINESS PATTERN DOSSIER that provides statistical evidence and 
        moving services context for the next agent to synthesize into business intelligence.

        INPUT DATA:
        - Table name: {table_name}
        - Glue metadata: {glue_metadata}
        - Sample data from Athena queries
        - Existing Weaviate knowledge for pattern comparison""",
        expected_output="""A detailed business pattern analysis dossier in JSON format containing:
        
        1. Statistical correlation findings with confidence levels
        2. Seasonal pattern analysis with business impact assessment
        3. Geographic variation insights with market implications  
        4. Moving services business relationships with operational impact
        5. Evidence-based recommendations for metadata synthesis
        
        The dossier should be data-driven, statistically validated, and rich with moving 
        services business context for optimal metadata generation.""",
        agent=pattern_discovery_profiler
    )

def create_business_synthesis_task() -> Task:
    """Create business synthesis task that outputs final YAML"""
    return Task(
        description="""Using the comprehensive business pattern dossier and draft YAML from the Scribe, 
        synthesize intelligent metadata that transforms technical data into actionable business intelligence.

        INPUT: Draft YAML from Scribe with basic technical structure
        OUTPUT: Complete YAML with AI-generated vectorized properties

        SYNTHESIS MISSION - AI-GENERATE ALL VECTORIZED PROPERTIES:
        1. DATASET METADATA (AI-Enhanced):
           - description: Transform basic table comment into rich business intelligence explanation
           - answerableQuestions: Generate JSON array of specific business questions this data answers
           - llmHints: Create JSON object with SQL generation guidance and business context

        2. COLUMN METADATA (AI-Enhanced):
           - description: Write business impact explanation for each column
           - businessName: Create user-friendly names that operations/finance teams recognize  
           - semanticType: Assign standardized categories for consistent analysis
           - sqlUsagePattern: Generate expert guidance for optimal query generation
           - sampleValues: Add representative values for context (from pattern analysis)

        MOVING SERVICES BUSINESS INTELLIGENCE FOCUS:
        Transform statistical evidence into business intelligence that serves:

        OPERATIONS MANAGERS:
        - Crew scheduling and efficiency optimization guidance
        - Truck utilization and route planning insights
        - Service quality and completion time factors

        FINANCE TEAMS:
        - Labor profitability calculation components
        - Pricing strategy and cost management data
        - Revenue optimization opportunities

        BUSINESS LEADERS:
        - Strategic value and competitive advantages
        - Market expansion decision support
        - Customer satisfaction and retention drivers

        CRITICAL: All vectorized properties (description, answerableQuestions) must be AI-generated
        with rich moving services business context. Keep technical structure but enhance with intelligence.

        Output the complete YAML exactly as it should be saved to the repo and ingested into Weaviate.""",
        expected_output="""Complete YAML file ready for Weaviate ingestion:

        ```yaml
        DatasetMetadata:
          tableName: [from_scribe]
          athenaTableName: [from_scribe] 
          description: "[AI-GENERATED rich business intelligence explanation]"
          answerableQuestions: '[AI-GENERATED JSON array of business questions]'
          llmHints: '[AI-GENERATED JSON object with SQL hints and business context]'
          dataOwner: "[INFERRED from business context]"
          sourceSystem: "[INFERRED from table patterns]"

        Column:
          - columnName: [from_scribe]
            athenaDataType: [from_scribe]
            parentAthenaTableName: [from_scribe]
            description: "[AI-GENERATED business impact explanation]"
            businessName: "[AI-GENERATED user-friendly name]"
            semanticType: "[AI-GENERATED standardized category]"
            sqlUsagePattern: "[AI-GENERATED query guidance]"
            sampleValues: [AI-DISCOVERED from pattern analysis]
            # Repeat for all columns
        ```

        YAML must be valid, complete, and ready for immediate Weaviate ingestion.""",
        agent=business_intelligence_synthesizer
    )

def create_validation_task() -> Task:
    """Create validation task for YAML output"""
    return Task(
        description="""Validate the complete YAML metadata for business intelligence quality, 
        technical structure, and moving services domain expertise.

        INPUT: Complete YAML ready for Weaviate ingestion
        OUTPUT: Validation decision with YAML quality assessment

        VALIDATION CRITERIA:

        1. YAML STRUCTURE VALIDATION:
           - Valid YAML syntax and format
           - All required fields present (tableName, athenaTableName, description, etc.)
           - Proper data types and JSON field formatting
           - Weaviate schema compatibility

        2. AI-GENERATED CONTENT QUALITY:
           - Are vectorized properties (description, answerableQuestions) AI-enhanced?
           - Do descriptions explain business VALUE, not just technical content?
           - Are answerableQuestions specific and business-relevant?
           - Do llmHints provide actionable SQL guidance?

        3. MOVING SERVICES BUSINESS INTELLIGENCE:
           - Does metadata demonstrate moving industry expertise?
           - Are labor cost, seasonal, and geographic factors addressed?
           - Do business names use operations/finance terminology?
           - Are semantic types appropriate for moving services analytics?

        4. STAKEHOLDER VALUE VERIFICATION:
           OPERATIONS MANAGERS: Crew scheduling, efficiency, service quality insights
           FINANCE TEAMS: Cost analysis, profitability, pricing guidance  
           BUSINESS LEADERS: Strategic opportunities, competitive advantages

        APPROVAL CRITERIA:
        - YAML is technically valid and Weaviate-ready
        - All vectorized properties are AI-enhanced with business intelligence
        - Demonstrates moving services domain expertise
        - Provides clear value to operations, finance, and leadership stakeholders

        If APPROVED: YAML is ready for Weaviate ingestion
        If REJECTED: Provide specific improvement recommendations""",
        expected_output="""YAML Validation Report in JSON format:
        
        {
          "yaml_validation_status": "APPROVED" or "REJECTED",
          "yaml_quality_score": 8.7,
          "technical_validation": {
            "yaml_syntax": "VALID",
            "required_fields": "COMPLETE", 
            "weaviate_compatibility": "READY"
          },
          "business_intelligence_assessment": {
            "vectorized_properties_enhanced": true,
            "moving_services_expertise": "COMPREHENSIVE",
            "stakeholder_value": "HIGH"
          },
          "ai_content_quality": {
            "descriptions_business_focused": true,
            "answerable_questions_relevant": true,
            "llm_hints_actionable": true
          },
          "approval_summary": "YAML demonstrates excellent business intelligence and is ready for Weaviate ingestion",
          "next_action": "PROCEED_TO_WEAVIATE_INGESTION" or "REQUIRES_IMPROVEMENT"
        }
        
        Only approve YAML that serves real moving services business intelligence needs.""",
        agent=quality_assurance_validator
    )

# Helper function to create all tasks with proper context
def create_metadata_tasks(table_name: str, glue_metadata: dict) -> tuple:
    """Create all tasks with proper context relationships"""
    
    # Create individual tasks
    pattern_task = create_pattern_discovery_task(table_name, glue_metadata)
    synthesis_task = create_business_synthesis_task()
    validation_task = create_validation_task()
    
    # Set up context relationships
    synthesis_task.context = [pattern_task]
    validation_task.context = [pattern_task, synthesis_task]
    
    return pattern_task, synthesis_task, validation_task