# tasks.py
"""
CrewAI tasks for moving services metadata generation
"""

from crewai import Task
from agents import (
    data_discovery_agent,
    business_intelligence_agent,
    metadata_quality_validator
)

# Task 1: Data Discovery
discover_data_patterns = Task(
    description="""Analyze {table_name} from the moving company's operational database to extract raw facts and patterns that support business analytics across all aspects of the full-spectrum relocation management platform.

DISCOVERY MISSION - Extract the Facts:
1. Use AthenaSamplingTool to get real data samples (100 rows minimum)
2. Use ColumnProfilerTool to analyze statistical patterns for key business columns
3. Use WeaviateSearchTool to find similar column patterns from existing metadata (if available)

MOVING SERVICES BUSINESS CONTEXT:
This national moving company operates as a comprehensive relocation management platform with:
- Sophisticated service portfolio across residential, commercial, and corporate segments
- Standardized process-driven service delivery with localized operational execution
- Tiered service levels from DIY support to premium white-glove offerings
- Specialized logistics capabilities for high-value and complex items
- Comprehensive asset protection protocols and quality management
- Integrated storage and ancillary services ecosystem

CRITICAL DATA PATTERNS TO IDENTIFY:
- Operational indicators: service delivery metrics, quality measures, efficiency tracking
- Financial metrics: revenue streams, cost structures, profitability calculations
- Geographic identifiers: location-based performance and market analysis
- Service classifications: tier levels, specialization areas, delivery methods
- Customer attributes: segmentation, preferences, engagement patterns
- Workflow tracking: process stages, status indicators, completion metrics

SAMPLE VALUES FOCUS:
Extract actual filterable values that enable business analytics across:
- Service optimization and operational efficiency
- Revenue analysis and financial performance
- Market analysis and geographic comparison
- Customer experience and satisfaction tracking
- Resource allocation and capacity planning

Your analysis must capture real business categories that support data-driven decision-making across all functional areas.""",
    expected_output="""Comprehensive data discovery report in JSON format containing:

{
  "table_analysis": {
    "table_name": "table_name_here",
    "total_columns": "number_here",
    "sample_rows_analyzed": 100,
    "moving_services_patterns_identified": [
      "operational_indicators_detected",
      "financial_metrics_found", 
      "geographic_identifiers_located",
      "service_classifications_identified"
    ]
  },
  "column_details": {
    "column_name_example": {
      "data_type": "athena_type_here",
      "sample_values": ["actual_values_from_data"],
      "statistics": {
        "total_rows": "count_here",
        "distinct_values": "count_here", 
        "null_percentage": "percentage_here",
        "cardinality_ratio": "ratio_here"
      },
      "business_pattern_analysis": "ai_interpretation_here",
      "similar_patterns_found": ["weaviate_matches_here"]
    }
  },
  "moving_services_insights": [
    "Operational columns: columns_here supporting service delivery and efficiency tracking",
    "Financial metrics: columns_here enabling revenue and profitability analysis", 
    "Geographic indicators: columns_here for location-based performance comparison",
    "Service classification: columns_here for tier and specialization analysis",
    "Customer attributes: columns_here for segmentation and experience tracking",
    "Workflow tracking: columns_here for process and completion monitoring"
  ],
  "business_analytics_supported": [
    "Specific analytics capabilities this data enables based on patterns found"
  ]
}""",
    agent=data_discovery_agent
)

# Task 2: Business Intelligence Generation
generate_business_metadata = Task(
    description="""Transform the data discovery findings into business-intelligent metadata that enables moving company stakeholders to answer critical business questions about operations, finance, and strategic performance.

INPUT: Data discovery report with real sample values and business patterns
OUTPUT: Complete YAML ready for Weaviate ingestion

SYNTHESIS MISSION - Generate AI-Enhanced Metadata:

FOR DATASETMETADATA (2 vectorized fields):
- description: Rich business intelligence explanation of what operational decisions this data enables
- answerableQuestions: JSON array of realistic moving industry questions this specific dataset can answer

FOR COLUMN METADATA (3 vectorized fields per column):
- description: Explain HOW this column contributes to business calculations and decisions
- businessName: Convert technical names to moving industry terminology
- semanticType: Assign standardized categories for consistent analytics

MOVING SERVICES BUSINESS INTELLIGENCE FOCUS:
Write metadata that serves three key stakeholders:

OPERATIONS MANAGERS need to know:
- How columns support service delivery optimization and quality management
- Which fields track efficiency and resource allocation metrics
- What data enables performance measurement and process improvement

FINANCE TEAMS need to understand:
- How columns contribute to revenue optimization and cost management
- Which fields support profitability analysis and pricing strategy
- What data enables financial performance tracking and margin analysis

EXECUTIVES need metadata that explains:
- How data supports strategic market analysis and competitive positioning
- Which fields enable performance benchmarking and growth planning
- What metrics drive business expansion and customer experience optimization

ANSWERABLE QUESTIONS EXAMPLES (variety, not specific assumptions):
- "How does performance vary across different operational dimensions?"
- "What are the key cost drivers and revenue patterns?"
- "Which locations or segments show the strongest performance?"
- "How do service delivery metrics correlate with business outcomes?"
- "What are the trends in customer engagement and satisfaction?"
- "Which operational areas present optimization opportunities?"

SEMANTIC TYPE CATEGORIES for moving services:
- operational_metric: service delivery, efficiency, quality, and performance measures
- financial_amount: revenue, costs, rates, and profitability indicators  
- geographic_identifier: location, market, and territory references
- workflow_status: process stages, completion states, and tracking indicators
- service_classification: tier levels, specialization areas, and delivery methods
- customer_attribute: demographics, preferences, engagement, and experience data

Use actual sample values from discovery for the sampleValues field.""",
    expected_output="""Complete YAML file ready for Weaviate ingestion:

```yaml
DatasetMetadata:
  tableName: table_name_from_input
  athenaTableName: full_table_name_from_input
  description: "[AI-generated business intelligence explanation focusing on moving services operations]"
  answerableQuestions: '[AI-generated JSON array of realistic business questions]'
  llmHints: '[AI-generated JSON object with query guidance and business context]'
  dataOwner: "[Inferred from business context - Operations Team/Finance Team/etc]"
  sourceSystem: "[Inferred from table patterns - Moving Management System/CRM/etc]"

Column:
  - columnName: "[from_discovery_data]"
    athenaDataType: "[from_discovery_data]"
    parentAthenaTableName: "[full_table_name_from_input]"
    description: "[AI-generated business impact explanation with moving services context]"
    businessName: "[AI-generated user-friendly name using industry terminology]"
    semanticType: "[AI-assigned standardized category]"
    sqlUsagePattern: "[AI-generated query guidance for analytics]"
    sampleValues: "[actual_values_from_discovery]"
    # Repeat for all columns discovered
```

CRITICAL: All vectorized properties must demonstrate moving services industry expertise and enable real business decision-making.""",
    agent=business_intelligence_agent,
    context=[discover_data_patterns]
)

# Task 3: Quality Validation
validate_metadata_quality = Task(
    description="""Validate the generated metadata meets the rigorous business intelligence standards required for moving company analytics and decision-making.

INPUT: Complete YAML metadata ready for Weaviate ingestion
OUTPUT: Validation decision with comprehensive quality assessment

VALIDATION CRITERIA - Four Critical Dimensions:

1. TECHNICAL YAML VALIDATION:
   - Valid YAML syntax and structure
   - All required schema fields present and properly formatted
   - JSON fields (answerableQuestions, llmHints) are valid JSON
   - Sample values are properly formatted arrays

2. MOVING SERVICES BUSINESS INTELLIGENCE:
   - Descriptions explain business VALUE and decision-making impact
   - Answerable questions reflect realistic moving industry analytics
   - Business names use terminology that operations/finance teams recognize
   - Semantic types enable proper analytics categorization

3. STAKEHOLDER VALUE VERIFICATION:
   OPERATIONS MANAGERS: Can they use this for service optimization and efficiency analysis?
   FINANCE TEAMS: Does this enable revenue analysis and cost management?
   EXECUTIVES: Will this support strategic market and performance analysis?

4. COMPREHENSIVE BUSINESS ANALYTICS SUPPORT:
   Validate metadata enables data-driven decision-making across:
   - Service delivery optimization and operational efficiency
   - Revenue analysis and financial performance tracking
   - Market analysis and geographic performance comparison
   - Customer experience and satisfaction measurement
   - Resource allocation and strategic planning

APPROVAL STANDARDS:
- APPROVED: Metadata demonstrates comprehensive moving services expertise and enables business analytics across all functional areas
- REJECTED: Metadata lacks business intelligence or fails to support decision-making across the relocation platform

SAMPLE VALUES VALIDATION:
Verify that sampleValues contain actual filterable business categories relevant to the table's purpose:
- Service delivery metrics should show operational performance indicators
- Financial data should show actual amounts, rates, or cost categories
- Geographic data should show actual location or market identifiers
- Status fields should show real workflow or process stage values
- Classification fields should show actual service or customer categories""",
    expected_output="""Comprehensive validation report in JSON format:

{
  "validation_status": "APPROVED or REJECTED",
  "overall_quality_score": 8.7,
  "technical_validation": {
    "yaml_syntax": "VALID or INVALID",
    "required_fields": "COMPLETE or INCOMPLETE", 
    "json_fields": "VALID or INVALID",
    "sample_values_format": "PROPER or IMPROPER"
  },
  "business_intelligence_assessment": {
    "moving_services_expertise": "COMPREHENSIVE or PARTIAL or MISSING",
    "stakeholder_value_score": 9.2,
    "decision_enablement": "HIGH or MEDIUM or LOW",
    "industry_terminology": "APPROPRIATE or NEEDS_IMPROVEMENT"
  },
  "critical_analytics_support": {
    "operational_analysis_enabled": true,
    "financial_analysis_enabled": true, 
    "geographic_analysis_enabled": true,
    "customer_analysis_enabled": true,
    "service_analysis_enabled": true
  },
  "sample_values_quality": {
    "filterable_for_analytics": true,
    "business_relevant_categories": true,
    "actual_data_values": true
  },
  "approval_summary": "[Explanation of approval/rejection decision]",
  "improvement_recommendations": [
    "[Specific suggestions if rejected or needs enhancement]"
  ],
  "next_action": "PROCEED_TO_WEAVIATE_INGESTION or REQUIRES_IMPROVEMENT"
}

Only approve metadata that enables real moving services business intelligence and decision-making.""",
    agent=metadata_quality_validator,
    context=[discover_data_patterns, generate_business_metadata]
)