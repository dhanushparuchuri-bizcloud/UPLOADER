# POC-3: Moving Services Metadata Generation Pipeline

A CrewAI-powered metadata generation system that transforms technical data schemas into actionable business intelligence for moving services companies.

## Overview

This project uses CrewAI agents with AWS Bedrock (Claude) to analyze data tables and generate rich metadata that serves operations managers, finance teams, and business leaders in the moving services industry.

## Features

- **🤖 AI-Powered Analysis**: Uses CrewAI agents with AWS Bedrock Claude models
- **📊 Statistical Pattern Discovery**: Analyzes correlations, seasonality, and geographic variations
- **🗄️ AWS Integration**: Works with Glue, Athena, and S3
- **🔍 Knowledge Search**: Integrates with Weaviate for existing metadata patterns
- **📝 YAML Output**: Generates structured metadata ready for ingestion
- **✅ Quality Validation**: Automated validation of business intelligence quality

## Architecture

### Agents
1. **Pattern Discovery Profiler**: Analyzes statistical patterns and business relationships
2. **Business Intelligence Synthesizer**: Transforms patterns into actionable metadata
3. **Quality Assurance Validator**: Validates business intelligence quality

### Tools
- **Athena Sampling Tool**: Samples and analyzes actual data
- **Correlation Discovery Tool**: Finds statistical relationships
- **Seasonality Detection Tool**: Identifies temporal patterns
- **Geographic Variation Tool**: Analyzes location-based differences
- **Weaviate Knowledge Search**: Searches existing metadata patterns

## Setup

### Prerequisites
- Python 3.12+
- AWS credentials with access to Glue, Athena, and Bedrock
- Weaviate instance (local or cloud)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dhanushparuchuri-bizcloud/UPLOADER.git
   cd UPLOADER
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv crewai_env
   source crewai_env/bin/activate  # On Windows: crewai_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file with:
   ```bash
   # AWS Configuration
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   BEDROCK_MODEL_ID_CLAUDE=anthropic.claude-3-5-sonnet-20241022-v2:0
   
   # Athena Configuration
   ATHENA_RESULTS_BUCKET=s3://your-athena-results-bucket/
   ATHENA_WORKGROUP=primary
   
   # Weaviate Configuration
   WEAVIATE_URL=http://localhost:8080
   ```

## Usage

### Test Connections
```bash
python run.py --database your_database --table your_table --test-connections
```

### Generate Metadata
```bash
python run.py --database your_database --table your_table
```

### Output Files
The system generates:
- `{database}_{table}_{timestamp}.yaml` - Metadata ready for Weaviate ingestion
- `{database}_{table}_{timestamp}_validation.json` - Quality validation results

## Project Structure

```
POC-3/
├── agents.py                 # CrewAI agent definitions
├── tasks.py                  # Task definitions for agents
├── run.py                    # Main execution script
├── requirements.txt          # Python dependencies
├── tools/                    # Analysis tools
│   ├── base_discovery_tools.py
│   ├── pattern_discovery_tools.py
│   ├── seasonality_geographic_tools.py
│   └── validation_tools.py
├── config/                   # Configuration files
│   ├── agents.yaml
│   └── tasks.yaml
└── output/                   # Generated files (gitignored)
```

## Key Components

### Pattern Discovery
- Statistical correlation analysis
- Seasonal pattern detection (moving industry specific)
- Geographic variation analysis
- Business workflow identification

### Business Intelligence
- Operations management insights
- Financial impact analysis
- Strategic decision support
- Cross-functional value identification

### Quality Validation
- YAML structure validation
- Business intelligence assessment
- Moving services domain expertise verification
- Stakeholder value confirmation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support, please open an issue in the GitHub repository. 