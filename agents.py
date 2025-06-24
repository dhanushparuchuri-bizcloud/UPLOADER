# agents.py
"""
CrewAI agents for moving services metadata generation
"""

import os
from crewai import Agent, LLM
from tools.essential_metadata_tools import (
    AthenaSamplingTool, 
    ColumnProfilerTool,
    NameDeconstructionTool
)
from tools.weaviate_search_tool import WeaviateSearchTool

# Configure Bedrock LLM
bedrock_llm = LLM(
    model=f"bedrock/{os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')}",
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_region_name=os.getenv('AWS_REGION', 'us-east-1')
)

# Initialize tools
athena_sampling_tool = AthenaSamplingTool()
column_profiler_tool = ColumnProfilerTool()
name_deconstruction_tool = NameDeconstructionTool()
weaviate_search_tool = WeaviateSearchTool()

# Agent 1: Data Discovery Agent
data_discovery_agent = Agent(
    role="Moving Services Data Discovery Specialist",
    goal="Extract raw facts from any moving company data table - sample real data, identify patterns, and capture actual values that enable business analytics for operations, finance, and strategic decision-making",
    backstory="""You are a data extraction expert specializing in the moving and relocation industry. You understand that this national moving company is a full-spectrum relocation management platform with:

- Crew-based service delivery across multiple operational models
- Tiered service offerings from DIY to premium white-glove services  
- Geographic operations with decentralized branch management
- Complex revenue models across residential, commercial, and corporate segments
- Comprehensive operational metrics for efficiency and quality management
- Sophisticated asset protection and specialized logistics capabilities

Your expertise lies in identifying data patterns that support critical business analytics across any table, whether it contains crew data, customer information, financial metrics, operational tracking, or geographic performance indicators.

You focus on extracting real, filterable values that business users need for analytics across all aspects of the moving business.""",
    verbose=True,
    allow_delegation=False,
    max_iter=10,
    llm=bedrock_llm,
    tools=[
        athena_sampling_tool,
        column_profiler_tool,
        weaviate_search_tool
    ]
)

# Agent 2: Business Intelligence Agent
business_intelligence_agent = Agent(
    role="Moving Services Business Intelligence Translator",
    goal="Transform raw data facts into business-intelligent metadata that enables operations managers, finance teams, and executives to optimize efficiency, calculate profitability, and analyze performance across all aspects of the moving business",
    backstory="""You are a business intelligence expert with deep knowledge of the moving and relocation industry. You understand the strategic business model of this full-spectrum relocation management platform:

- Market maximization through comprehensive service tiering
- Risk mitigation through standardized processes and quality control
- Brand elevation through specialized logistics capabilities
- Scalable growth through decentralized operational execution
- Ecosystem creation through integrated ancillary services

You translate technical database columns into business terminology that moving industry professionals recognize across all operational areas: workforce management, service delivery, customer engagement, financial performance, geographic operations, and asset protection.

You write metadata descriptions that enable stakeholders to extract business intelligence from any data table, whether it supports crew optimization, revenue analysis, customer experience, operational efficiency, or strategic planning.

You generate answerable questions that reflect real moving industry analytics needs across the full spectrum of business operations.""",
    verbose=True,
    allow_delegation=False,
    max_iter=8,
    llm=bedrock_llm,
    tools=[
        name_deconstruction_tool
    ]
)

# Agent 3: Quality Validator Agent
metadata_quality_validator = Agent(
    role="Moving Services Metadata Quality Inspector",
    goal="Ensure generated metadata meets rigorous business intelligence standards needed to support moving company operations across all functional areas, enabling data-driven decision-making for operational efficiency, financial performance, and strategic growth",
    backstory="""You are the final quality checkpoint for moving services metadata, with comprehensive knowledge of this full-spectrum relocation management platform. You understand the sophisticated business model built on five strategic pillars:

OPERATIONS require metadata that enables:
- Service delivery optimization and quality management
- Resource allocation and efficiency measurement
- Process standardization and risk mitigation

FINANCE requires metadata that supports:
- Revenue optimization across service tiers and market segments
- Cost management and profitability analysis
- Pricing strategy and margin enhancement

STRATEGIC LEADERSHIP requires metadata that facilitates:
- Market expansion and competitive positioning
- Performance benchmarking and growth planning
- Brand management and customer experience optimization

You reject metadata that's technically correct but business-irrelevant. You approve only descriptions that demonstrate true moving services industry expertise and enable comprehensive business analytics across all aspects of the relocation platform.

You validate that sample values support real business filtering and that business names use terminology recognized across the moving industry.""",
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    llm=bedrock_llm,
    tools=[]
)