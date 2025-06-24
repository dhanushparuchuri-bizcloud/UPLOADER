"""
Validation tools for business intelligence quality assurance
"""

import json
import yaml
import logging
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

class YAMLValidatorTool(BaseTool):
    name: str = "Metadata Structure Validator"
    description: str = "Validate YAML structure and technical requirements for Weaviate ingestion"
    
    def _run(self, yaml_content: str) -> str:
        """Validate YAML structure and technical requirements"""
        try:
            logger.info("🔍 Validating YAML structure and technical requirements")
            
            validation_results = {
                "structure_validation": self._validate_yaml_structure(yaml_content),
                "required_fields_check": self._check_required_fields(yaml_content),
                "data_types_validation": self._validate_data_types(yaml_content),
                "weaviate_compatibility": self._check_weaviate_compatibility(yaml_content),
                "json_fields_validation": self._validate_json_fields(yaml_content)
            }
            
            # Generate overall validation report
            overall_status = self._determine_overall_status(validation_results)
            
            return json.dumps({
                "validation_status": overall_status,
                "detailed_results": validation_results,
                "validation_summary": self._generate_validation_summary(validation_results),
                "technical_recommendations": self._generate_technical_recommendations(validation_results)
            }, indent=2)
            
        except Exception as e:
            logger.error(f"YAML validation failed: {e}")
            return json.dumps({
                "validation_status": "ERROR",
                "error": str(e),
                "recommendation": "Fix YAML structure errors before proceeding"
            })
    
    def _validate_yaml_structure(self, yaml_content: str) -> Dict:
        """Validate basic YAML structure"""
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
            
            if not isinstance(parsed_yaml, dict):
                return {
                    "status": "FAIL",
                    "error": "YAML content is not a valid dictionary structure"
                }
            
            # Check for required top-level sections
            required_sections = ["DatasetMetadata", "Column"]
            missing_sections = [section for section in required_sections if section not in parsed_yaml]
            
            if missing_sections:
                return {
                    "status": "FAIL",
                    "error": f"Missing required sections: {missing_sections}",
                    "found_sections": list(parsed_yaml.keys())
                }
            
            return {
                "status": "PASS",
                "sections_found": list(parsed_yaml.keys()),
                "structure_valid": True
            }
            
        except yaml.YAMLError as e:
            return {
                "status": "FAIL",
                "error": f"Invalid YAML syntax: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "error": f"YAML parsing error: {str(e)}"
            }
    
    def _check_required_fields(self, yaml_content: str) -> Dict:
        """Check for required fields in each section"""
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
            
            # Required fields for DatasetMetadata
            dataset_required_fields = [
                "tableName", "athenaTableName", "description", 
                "answerableQuestions", "dataOwner", "sourceSystem", "llmHints"
            ]
            
            # Required fields for Column (assuming it's a list)
            column_required_fields = [
                "columnName", "athenaDataType", "parentAthenaTableName", 
                "description", "businessName", "semanticType", "sqlUsagePattern"
            ]
            
            results = {}
            
            # Check DatasetMetadata
            if "DatasetMetadata" in parsed_yaml:
                dataset_data = parsed_yaml["DatasetMetadata"]
                missing_dataset_fields = [
                    field for field in dataset_required_fields 
                    if field not in dataset_data or not dataset_data[field]
                ]
                
                results["DatasetMetadata"] = {
                    "status": "PASS" if not missing_dataset_fields else "FAIL",
                    "missing_fields": missing_dataset_fields,
                    "present_fields": [f for f in dataset_required_fields if f in dataset_data]
                }
            
            # Check Column data
            if "Column" in parsed_yaml:
                column_data = parsed_yaml["Column"]
                
                if isinstance(column_data, list):
                    column_results = []
                    for i, col in enumerate(column_data):
                        missing_col_fields = [
                            field for field in column_required_fields 
                            if field not in col or not col[field]
                        ]
                        
                        column_results.append({
                            "column_index": i,
                            "column_name": col.get("columnName", f"Column_{i}"),
                            "status": "PASS" if not missing_col_fields else "FAIL",
                            "missing_fields": missing_col_fields
                        })
                    
                    results["Column"] = {
                        "status": "PASS" if all(c["status"] == "PASS" for c in column_results) else "FAIL",
                        "columns_checked": len(column_results),
                        "column_details": column_results
                    }
                
                elif isinstance(column_data, dict):
                    # Single column object
                    missing_col_fields = [
                        field for field in column_required_fields 
                        if field not in column_data or not column_data[field]
                    ]
                    
                    results["Column"] = {
                        "status": "PASS" if not missing_col_fields else "FAIL",
                        "missing_fields": missing_col_fields,
                        "column_name": column_data.get("columnName", "Unknown")
                    }
            
            return results
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Required fields check failed: {str(e)}"
            }
    
    def _validate_data_types(self, yaml_content: str) -> Dict:
        """Validate data types of fields"""
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
            results = {}
            
            # DatasetMetadata data type checks
            if "DatasetMetadata" in parsed_yaml:
                dataset = parsed_yaml["DatasetMetadata"]
                dataset_validations = []
                
                # String fields
                string_fields = ["tableName", "athenaTableName", "description", "dataOwner", "sourceSystem"]
                for field in string_fields:
                    if field in dataset:
                        if not isinstance(dataset[field], str):
                            dataset_validations.append(f"{field} should be string, got {type(dataset[field])}")
                
                # JSON string fields
                json_fields = ["answerableQuestions", "llmHints"]
                for field in json_fields:
                    if field in dataset:
                        if not isinstance(dataset[field], str):
                            dataset_validations.append(f"{field} should be JSON string, got {type(dataset[field])}")
                
                results["DatasetMetadata"] = {
                    "status": "PASS" if not dataset_validations else "FAIL",
                    "validation_errors": dataset_validations
                }
            
            # Column data type checks
            if "Column" in parsed_yaml:
                column_data = parsed_yaml["Column"]
                column_validations = []
                
                columns_to_check = column_data if isinstance(column_data, list) else [column_data]
                
                for i, col in enumerate(columns_to_check):
                    col_errors = []
                    
                    # String fields
                    string_fields = ["columnName", "athenaDataType", "parentAthenaTableName", 
                                   "description", "businessName", "semanticType", "sqlUsagePattern"]
                    
                    for field in string_fields:
                        if field in col and not isinstance(col[field], str):
                            col_errors.append(f"{field} should be string, got {type(col[field])}")
                    
                    # List fields
                    if "sampleValues" in col and not isinstance(col["sampleValues"], list):
                        col_errors.append(f"sampleValues should be list, got {type(col['sampleValues'])}")
                    
                    # Boolean fields
                    boolean_fields = ["isPrimaryKey", "isForeignKey"]
                    for field in boolean_fields:
                        if field in col and not isinstance(col[field], bool):
                            col_errors.append(f"{field} should be boolean, got {type(col[field])}")
                    
                    if col_errors:
                        column_validations.append({
                            "column_index": i,
                            "column_name": col.get("columnName", f"Column_{i}"),
                            "errors": col_errors
                        })
                
                results["Column"] = {
                    "status": "PASS" if not column_validations else "FAIL",
                    "validation_errors": column_validations
                }
            
            return results
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Data type validation failed: {str(e)}"
            }
    
    def _check_weaviate_compatibility(self, yaml_content: str) -> Dict:
        """Check compatibility with Weaviate schema requirements"""
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
            compatibility_issues = []
            
            # Check for reserved Weaviate property names
            reserved_names = ["id", "_additional", "vector"]
            
            if "Column" in parsed_yaml:
                column_data = parsed_yaml["Column"]
                columns_to_check = column_data if isinstance(column_data, list) else [column_data]
                
                for col in columns_to_check:
                    col_name = col.get("columnName", "")
                    if col_name.lower() in reserved_names:
                        compatibility_issues.append(
                            f"Column name '{col_name}' conflicts with Weaviate reserved property"
                        )
                    
                    # Check for valid property naming (no spaces, special chars)
                    if col_name and not col_name.replace("_", "").replace("-", "").isalnum():
                        compatibility_issues.append(
                            f"Column name '{col_name}' contains invalid characters for Weaviate"
                        )
            
            # Check athenaTableName format
            if "DatasetMetadata" in parsed_yaml:
                athena_table = parsed_yaml["DatasetMetadata"].get("athenaTableName", "")
                if athena_table and "." not in athena_table:
                    compatibility_issues.append(
                        "athenaTableName should include database prefix (e.g., 'database.table')"
                    )
            
            return {
                "status": "PASS" if not compatibility_issues else "FAIL",
                "compatibility_issues": compatibility_issues,
                "recommendations": self._generate_compatibility_recommendations(compatibility_issues)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Weaviate compatibility check failed: {str(e)}"
            }
    
    def _validate_json_fields(self, yaml_content: str) -> Dict:
        """Validate JSON fields can be parsed"""
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
            json_validation_results = []
            
            # Check DatasetMetadata JSON fields
            if "DatasetMetadata" in parsed_yaml:
                dataset = parsed_yaml["DatasetMetadata"]
                
                json_fields = {
                    "answerableQuestions": "array",
                    "llmHints": "object"
                }
                
                for field, expected_type in json_fields.items():
                    if field in dataset:
                        try:
                            parsed_json = json.loads(dataset[field])
                            
                            # Validate expected type
                            if expected_type == "array" and not isinstance(parsed_json, list):
                                json_validation_results.append({
                                    "field": field,
                                    "status": "FAIL",
                                    "error": f"Expected array, got {type(parsed_json)}"
                                })
                            elif expected_type == "object" and not isinstance(parsed_json, dict):
                                json_validation_results.append({
                                    "field": field,
                                    "status": "FAIL", 
                                    "error": f"Expected object, got {type(parsed_json)}"
                                })
                            else:
                                json_validation_results.append({
                                    "field": field,
                                    "status": "PASS",
                                    "parsed_type": type(parsed_json).__name__
                                })
                                
                        except json.JSONDecodeError as e:
                            json_validation_results.append({
                                "field": field,
                                "status": "FAIL",
                                "error": f"Invalid JSON: {str(e)}"
                            })
            
            return {
                "status": "PASS" if all(r["status"] == "PASS" for r in json_validation_results) else "FAIL",
                "json_field_results": json_validation_results
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"JSON validation failed: {str(e)}"
            }
    
    def _determine_overall_status(self, validation_results: Dict) -> str:
        """Determine overall validation status"""
        all_checks = []
        
        for check_name, check_result in validation_results.items():
            if isinstance(check_result, dict) and "status" in check_result:
                all_checks.append(check_result["status"])
            elif isinstance(check_result, dict):
                # Handle nested results
                for sub_check in check_result.values():
                    if isinstance(sub_check, dict) and "status" in sub_check:
                        all_checks.append(sub_check["status"])
        
        if any(status == "ERROR" for status in all_checks):
            return "ERROR"
        elif any(status == "FAIL" for status in all_checks):
            return "FAIL"
        else:
            return "PASS"
    
    def _generate_validation_summary(self, validation_results: Dict) -> Dict:
        """Generate validation summary"""
        summary = {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "error_checks": 0
        }
        
        for check_result in validation_results.values():
            if isinstance(check_result, dict):
                if "status" in check_result:
                    summary["total_checks"] += 1
                    if check_result["status"] == "PASS":
                        summary["passed_checks"] += 1
                    elif check_result["status"] == "FAIL":
                        summary["failed_checks"] += 1
                    elif check_result["status"] == "ERROR":
                        summary["error_checks"] += 1
                else:
                    # Handle nested results
                    for sub_result in check_result.values():
                        if isinstance(sub_result, dict) and "status" in sub_result:
                            summary["total_checks"] += 1
                            if sub_result["status"] == "PASS":
                                summary["passed_checks"] += 1
                            elif sub_result["status"] == "FAIL":
                                summary["failed_checks"] += 1
                            elif sub_result["status"] == "ERROR":
                                summary["error_checks"] += 1
        
        return summary
    
    def _generate_technical_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate technical recommendations based on validation results"""
        recommendations = []
        
        # Structure recommendations
        structure_result = validation_results.get("structure_validation", {})
        if structure_result.get("status") == "FAIL":
            recommendations.append("Fix YAML syntax errors before proceeding")
        
        # Required fields recommendations
        required_fields = validation_results.get("required_fields_check", {})
        for section, section_result in required_fields.items():
            if isinstance(section_result, dict) and section_result.get("status") == "FAIL":
                missing = section_result.get("missing_fields", [])
                if missing:
                    recommendations.append(f"Add missing required fields in {section}: {', '.join(missing)}")
        
        # Data type recommendations
        data_types = validation_results.get("data_types_validation", {})
        for section, section_result in data_types.items():
            if isinstance(section_result, dict) and section_result.get("status") == "FAIL":
                recommendations.append(f"Fix data type errors in {section}")
        
        # Weaviate compatibility recommendations
        weaviate_compat = validation_results.get("weaviate_compatibility", {})
        if weaviate_compat.get("status") == "FAIL":
            recommendations.extend(weaviate_compat.get("recommendations", []))
        
        # JSON validation recommendations
        json_validation = validation_results.get("json_fields_validation", {})
        if json_validation.get("status") == "FAIL":
            recommendations.append("Fix JSON syntax in structured fields")
        
        return recommendations
    
    def _generate_compatibility_recommendations(self, issues: List[str]) -> List[str]:
        """Generate Weaviate compatibility recommendations"""
        recommendations = []
        
        for issue in issues:
            if "reserved property" in issue:
                recommendations.append("Rename columns that conflict with Weaviate reserved properties")
            elif "invalid characters" in issue:
                recommendations.append("Use alphanumeric characters and underscores only in column names")
            elif "database prefix" in issue:
                recommendations.append("Add database prefix to athenaTableName (e.g., 'database.table')")
        
        return recommendations


class BusinessLogicValidatorTool(BaseTool):
    name: str = "Business Intelligence Quality Validator"
    description: str = "Validate business intelligence quality and stakeholder value for moving services"
    
    def _run(self, metadata_content: str) -> str:
        """Validate business intelligence quality and stakeholder value"""
        try:
            logger.info("🧠 Validating business intelligence quality")
            
            # Parse metadata content
            try:
                if metadata_content.strip().startswith('{'):
                    metadata = json.loads(metadata_content)
                else:
                    metadata = yaml.safe_load(metadata_content)
            except Exception as e:
                return json.dumps({
                    "validation_status": "ERROR",
                    "error": f"Could not parse metadata content: {str(e)}"
                })
            
            business_validation = {
                "stakeholder_value": self._assess_stakeholder_value(metadata),
                "business_impact_clarity": self._check_business_impact(metadata),
                "moving_services_context": self._validate_moving_services_context(metadata),
                "actionability_assessment": self._assess_actionability(metadata),
                "business_terminology": self._validate_business_terminology(metadata)
            }
            
            # Generate overall business intelligence score
            bi_score = self._calculate_business_intelligence_score(business_validation)
            
            return json.dumps({
                "business_intelligence_score": bi_score,
                "validation_details": business_validation,
                "approval_recommendation": self._generate_approval_recommendation(business_validation, bi_score),
                "improvement_suggestions": self._generate_improvement_suggestions(business_validation)
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Business logic validation failed: {e}")
            return json.dumps({
                "validation_status": "ERROR",
                "error": str(e)
            })
    
    def _assess_stakeholder_value(self, metadata: Dict) -> Dict:
        """Assess value for different moving services stakeholders"""
        stakeholder_assessments = {
            "operations_manager": self._assess_operations_value(metadata),
            "finance_team": self._assess_finance_value(metadata),
            "business_leader": self._assess_strategic_value(metadata)
        }
        
        return {
            "individual_assessments": stakeholder_assessments,
            "overall_stakeholder_value": self._calculate_overall_stakeholder_value(stakeholder_assessments)
        }
    
    def _assess_operations_value(self, metadata: Dict) -> Dict:
        """Assess value for operations managers"""
        operations_keywords = [
            "crew", "efficiency", "productivity", "scheduling", "completion", 
            "utilization", "optimization", "service quality", "operational"
        ]
        
        description_text = self._extract_description_text(metadata).lower()
        business_names = self._extract_business_names(metadata)
        
        # Count operations-related keywords
        keyword_matches = sum(1 for keyword in operations_keywords if keyword in description_text)
        
        # Check for operations-specific value propositions
        operations_indicators = [
            "affects crew scheduling" in description_text,
            "impacts efficiency" in description_text,
            "operational" in description_text,
            "productivity" in description_text,
            any("crew" in name.lower() for name in business_names),
            any("efficiency" in name.lower() for name in business_names)
        ]
        
        value_score = (keyword_matches / len(operations_keywords)) * 0.6 + \
                     (sum(operations_indicators) / len(operations_indicators)) * 0.4
        
        return {
            "value_score": min(1.0, value_score),
            "keyword_matches": keyword_matches,
            "operations_indicators_found": sum(operations_indicators),
            "assessment": "high" if value_score > 0.7 else "medium" if value_score > 0.4 else "low",
            "specific_value": self._identify_specific_operations_value(description_text)
        }
    
    def _assess_finance_value(self, metadata: Dict) -> Dict:
        """Assess value for finance teams"""
        finance_keywords = [
            "cost", "revenue", "profit", "margin", "financial", "pricing", 
            "expense", "budget", "profitability", "calculation"
        ]
        
        description_text = self._extract_description_text(metadata).lower()
        business_names = self._extract_business_names(metadata)
        
        # Count finance-related keywords
        keyword_matches = sum(1 for keyword in finance_keywords if keyword in description_text)
        
        # Check for finance-specific value propositions
        finance_indicators = [
            "cost" in description_text,
            "revenue" in description_text,
            "profitability" in description_text,
            "financial" in description_text,
            any("cost" in name.lower() for name in business_names),
            any("revenue" in name.lower() for name in business_names)
        ]
        
        value_score = (keyword_matches / len(finance_keywords)) * 0.6 + \
                     (sum(finance_indicators) / len(finance_indicators)) * 0.4
        
        return {
            "value_score": min(1.0, value_score),
            "keyword_matches": keyword_matches,
            "finance_indicators_found": sum(finance_indicators),
            "assessment": "high" if value_score > 0.7 else "medium" if value_score > 0.4 else "low",
            "specific_value": self._identify_specific_finance_value(description_text)
        }
    
    def _assess_strategic_value(self, metadata: Dict) -> Dict:
        """Assess value for business leaders"""
        strategic_keywords = [
            "strategic", "competitive", "market", "growth", "opportunity", 
            "performance", "customer satisfaction", "business", "advantage"
        ]
        
        description_text = self._extract_description_text(metadata).lower()
        answerable_questions = self._extract_answerable_questions(metadata)
        
        # Count strategic keywords
        keyword_matches = sum(1 for keyword in strategic_keywords if keyword in description_text)
        
        # Check for strategic value indicators
        strategic_indicators = [
            "strategic" in description_text,
            "competitive" in description_text,
            "market" in description_text,
            "customer satisfaction" in description_text,
            len(answerable_questions) > 0,
            any("optimize" in q.lower() for q in answerable_questions),
            any("performance" in q.lower() for q in answerable_questions)
        ]
        
        value_score = (keyword_matches / len(strategic_keywords)) * 0.5 + \
                     (sum(strategic_indicators) / len(strategic_indicators)) * 0.5
        
        return {
            "value_score": min(1.0, value_score),
            "keyword_matches": keyword_matches,
            "strategic_indicators_found": sum(strategic_indicators),
            "assessment": "high" if value_score > 0.7 else "medium" if value_score > 0.4 else "low",
            "specific_value": self._identify_specific_strategic_value(description_text, answerable_questions)
        }
    
    def _check_business_impact(self, metadata: Dict) -> Dict:
        """Check clarity of business impact explanations"""
        description_text = self._extract_description_text(metadata)
        
        impact_indicators = {
            "explains_why": any(phrase in description_text.lower() for phrase in [
                "affects", "impacts", "influences", "drives", "enables", "supports"
            ]),
            "mentions_outcomes": any(phrase in description_text.lower() for phrase in [
                "efficiency", "cost", "revenue", "performance", "satisfaction", "profitability"
            ]),
            "provides_context": any(phrase in description_text.lower() for phrase in [
                "because", "since", "due to", "resulting in", "leading to"
            ]),
            "actionable_insights": any(phrase in description_text.lower() for phrase in [
                "optimize", "improve", "reduce", "increase", "enhance"
            ])
        }
        
        clarity_score = sum(impact_indicators.values()) / len(impact_indicators)
        
        return {
            "clarity_score": clarity_score,
            "impact_indicators": impact_indicators,
            "assessment": "excellent" if clarity_score >= 0.8 else "good" if clarity_score >= 0.6 else "poor",
            "missing_elements": [k for k, v in impact_indicators.items() if not v]
        }
    
    def _validate_moving_services_context(self, metadata: Dict) -> Dict:
        """Validate moving services domain expertise"""
        moving_industry_terms = [
            "crew", "labor", "moving", "truck", "vehicle", "branch", "location",
            "customer", "move", "delivery", "service", "equipment", "operational"
        ]
        
        business_concepts = {
            "labor_intensive": ["crew", "labor", "hourly", "staff", "worker", "employee"],
            "seasonal_business": ["season", "peak", "summer", "winter", "demand", "volume"],
            "geographic_operations": ["branch", "location", "region", "market", "geographic"],
            "customer_service": ["satisfaction", "quality", "delivery", "completion", "service"],
            "equipment_utilization": ["truck", "vehicle", "equipment", "utilization", "fleet"]
        }
        
        description_text = self._extract_description_text(metadata).lower()
        
        # Check for moving industry terminology
        industry_term_matches = sum(1 for term in moving_industry_terms if term in description_text)
        
        # Check for business concept understanding
        concept_scores = {}
        for concept, keywords in business_concepts.items():
            concept_matches = sum(1 for keyword in keywords if keyword in description_text)
            concept_scores[concept] = concept_matches > 0
        
        expertise_score = (industry_term_matches / len(moving_industry_terms)) * 0.6 + \
                         (sum(concept_scores.values()) / len(concept_scores)) * 0.4
        
        return {
            "expertise_score": min(1.0, expertise_score),
            "industry_terms_found": industry_term_matches,
            "business_concepts_demonstrated": concept_scores,
            "assessment": "comprehensive" if expertise_score >= 0.7 else "partial" if expertise_score >= 0.4 else "missing",
            "domain_evidence": self._identify_domain_evidence(description_text)
        }
    
    def _assess_actionability(self, metadata: Dict) -> Dict:
        """Assess how actionable the metadata is for business decisions"""
        description_text = self._extract_description_text(metadata)
        answerable_questions = self._extract_answerable_questions(metadata)
        llm_hints = self._extract_llm_hints(metadata)
        
        actionability_indicators = {
            "specific_use_cases": len(answerable_questions) > 0,
            "optimization_guidance": any(word in description_text.lower() for word in [
                "optimize", "improve", "reduce", "increase", "enhance", "maximize"
            ]),
            "decision_support": any(phrase in description_text.lower() for phrase in [
                "decide", "choose", "determine", "evaluate", "compare", "assess"
            ]),
            "business_questions": any("?" in q for q in answerable_questions),
            "implementation_hints": isinstance(llm_hints, dict) and len(llm_hints) > 0
        }
        
        actionability_score = sum(actionability_indicators.values()) / len(actionability_indicators)
        
        return {
            "actionability_score": actionability_score,
            "actionability_indicators": actionability_indicators,
            "assessment": "highly_actionable" if actionability_score >= 0.8 else "moderately_actionable" if actionability_score >= 0.6 else "limited_actionability",
            "business_questions_count": len(answerable_questions)
        }
    
    def _validate_business_terminology(self, metadata: Dict) -> Dict:
        """Validate use of appropriate business terminology"""
        business_names = self._extract_business_names(metadata)
        description_text = self._extract_description_text(metadata)
        
        # Check for user-friendly business names
        business_name_quality = []
        for name in business_names:
            if name:
                quality_score = 0
                # Avoid technical jargon
                if not any(tech_word in name.lower() for tech_word in ["varchar", "bigint", "decimal", "timestamp"]):
                    quality_score += 0.3
                # Use business-friendly terms
                if any(biz_word in name.lower() for biz_word in ["rate", "count", "total", "average", "score"]):
                    quality_score += 0.3
                # Proper capitalization/formatting
                if name.replace(" ", "").replace("_", "").isalnum():
                    quality_score += 0.2
                # Descriptive length
                if 10 <= len(name) <= 50:
                    quality_score += 0.2
                
                business_name_quality.append(quality_score)
        
        avg_business_name_quality = sum(business_name_quality) / len(business_name_quality) if business_name_quality else 0
        
        # Check description terminology
        avoids_jargon = not any(jargon in description_text.lower() for jargon in [
            "varchar", "bigint", "decimal", "nullable", "primary key", "foreign key"
        ])
        
        uses_business_language = any(biz_term in description_text.lower() for biz_term in [
            "business", "operation", "customer", "service", "performance", "efficiency"
        ])
        
        terminology_score = (avg_business_name_quality * 0.5) + \
                           (0.3 if avoids_jargon else 0) + \
                           (0.2 if uses_business_language else 0)
        
        return {
            "terminology_score": terminology_score,
            "business_name_quality": avg_business_name_quality,
            "avoids_technical_jargon": avoids_jargon,
            "uses_business_language": uses_business_language,
            "assessment": "excellent" if terminology_score >= 0.8 else "good" if terminology_score >= 0.6 else "needs_improvement"
        }
    
    def _calculate_business_intelligence_score(self, validation_results: Dict) -> float:
        """Calculate overall business intelligence score"""
        scores = []
        
        # Stakeholder value (30% weight)
        stakeholder_value = validation_results.get("stakeholder_value", {}).get("overall_stakeholder_value", 0)
        scores.append(stakeholder_value * 0.3)
        
        # Business impact clarity (25% weight)
        business_impact = validation_results.get("business_impact_clarity", {}).get("clarity_score", 0)
        scores.append(business_impact * 0.25)
        
        # Moving services context (20% weight)
        moving_context = validation_results.get("moving_services_context", {}).get("expertise_score", 0)
        scores.append(moving_context * 0.2)
        
        # Actionability (15% weight)
        actionability = validation_results.get("actionability_assessment", {}).get("actionability_score", 0)
        scores.append(actionability * 0.15)
        
        # Business terminology (10% weight)
        terminology = validation_results.get("business_terminology", {}).get("terminology_score", 0)
        scores.append(terminology * 0.1)
        
        return sum(scores)
    
    def _calculate_overall_stakeholder_value(self, assessments: Dict) -> float:
        """Calculate overall stakeholder value score"""
        scores = [assessment.get("value_score", 0) for assessment in assessments.values()]
        return sum(scores) / len(scores) if scores else 0
    
    def _extract_description_text(self, metadata: Dict) -> str:
        """Extract all description text from metadata"""
        descriptions = []
        
        # DatasetMetadata description
        if "DatasetMetadata" in metadata:
            desc = metadata["DatasetMetadata"].get("description", "")
            if desc:
                descriptions.append(desc)
        
        # Column descriptions
        if "Column" in metadata:
            column_data = metadata["Column"]
            columns = column_data if isinstance(column_data, list) else [column_data]
            
            for col in columns:
                desc = col.get("description", "")
                if desc:
                    descriptions.append(desc)
        
        return " ".join(descriptions)
    
    def _extract_business_names(self, metadata: Dict) -> List[str]:
        """Extract business names from metadata"""
        business_names = []
        
        if "Column" in metadata:
            column_data = metadata["Column"]
            columns = column_data if isinstance(column_data, list) else [column_data]
            
            for col in columns:
                business_name = col.get("businessName", "")
                if business_name:
                    business_names.append(business_name)
        
        return business_names
    
    def _extract_answerable_questions(self, metadata: Dict) -> List[str]:
        """Extract answerable questions from metadata"""
        questions = []
        
        if "DatasetMetadata" in metadata:
            questions_json = metadata["DatasetMetadata"].get("answerableQuestions", "[]")
            try:
                questions = json.loads(questions_json) if isinstance(questions_json, str) else questions_json
            except json.JSONDecodeError:
                pass
        
        return questions if isinstance(questions, list) else []
    
    def _extract_llm_hints(self, metadata: Dict) -> Dict:
        """Extract LLM hints from metadata"""
        hints = {}
        
        if "DatasetMetadata" in metadata:
            hints_json = metadata["DatasetMetadata"].get("llmHints", "{}")
            try:
                hints = json.loads(hints_json) if isinstance(hints_json, str) else hints_json
            except json.JSONDecodeError:
                pass
        
        return hints if isinstance(hints, dict) else {}
    
    def _identify_specific_operations_value(self, description: str) -> List[str]:
        """Identify specific operations value propositions"""
        value_props = []
        
        if "crew" in description:
            value_props.append("Crew management and scheduling optimization")
        if "efficiency" in description:
            value_props.append("Operational efficiency measurement and improvement")
        if "utilization" in description:
            value_props.append("Resource utilization optimization")
        if "completion" in description:
            value_props.append("Service completion tracking and analysis")
        
        return value_props
    
    def _identify_specific_finance_value(self, description: str) -> List[str]:
        """Identify specific finance value propositions"""
        value_props = []
        
        if "cost" in description:
            value_props.append("Cost analysis and management")
        if "revenue" in description:
            value_props.append("Revenue tracking and optimization")
        if "profit" in description:
            value_props.append("Profitability analysis")
        if "pricing" in description:
            value_props.append("Pricing strategy support")
        
        return value_props
    
    def _identify_specific_strategic_value(self, description: str, questions: List[str]) -> List[str]:
        """Identify specific strategic value propositions"""
        value_props = []
        
        if "competitive" in description:
            value_props.append("Competitive analysis and positioning")
        if "market" in description:
            value_props.append("Market analysis and expansion opportunities")
        if "customer satisfaction" in description:
            value_props.append("Customer experience optimization")
        if any("performance" in q.lower() for q in questions):
            value_props.append("Performance measurement and benchmarking")
        
        return value_props
    
    def _identify_domain_evidence(self, description: str) -> List[str]:
        """Identify evidence of moving services domain expertise"""
        evidence = []
        
        if "crew" in description and "labor" in description:
            evidence.append("Understands labor-intensive operations")
        if any(season in description for season in ["season", "peak", "summer"]):
            evidence.append("Recognizes seasonal business patterns")
        if "branch" in description or "location" in description:
            evidence.append("Acknowledges geographic operations")
        if "customer" in description and "satisfaction" in description:
            evidence.append("Identifies customer satisfaction drivers")
        
        return evidence
    
    def _generate_approval_recommendation(self, validation_results: Dict, bi_score: float) -> Dict:
        """Generate approval recommendation based on validation results"""
        if bi_score >= 0.8:
            return {
                "recommendation": "APPROVE",
                "confidence": "high",
                "reasoning": "Metadata demonstrates excellent business intelligence quality with strong stakeholder value"
            }
        elif bi_score >= 0.6:
            return {
                "recommendation": "APPROVE_WITH_MINOR_IMPROVEMENTS",
                "confidence": "medium",
                "reasoning": "Metadata shows good business intelligence but could benefit from minor enhancements"
            }
        else:
            return {
                "recommendation": "REJECT",
                "confidence": "high",
                "reasoning": "Metadata lacks sufficient business intelligence and stakeholder value"
            }
    
    def _generate_improvement_suggestions(self, validation_results: Dict) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Stakeholder value improvements
        stakeholder_value = validation_results.get("stakeholder_value", {}).get("individual_assessments", {})
        
        for stakeholder, assessment in stakeholder_value.items():
            if assessment.get("assessment") == "low":
                suggestions.append(f"Enhance value proposition for {stakeholder.replace('_', ' ')} - explain specific benefits")
        
        # Business impact improvements
        business_impact = validation_results.get("business_impact_clarity", {})
        missing_elements = business_impact.get("missing_elements", [])
        
        for element in missing_elements:
            if element == "explains_why":
                suggestions.append("Add explanations of WHY this data matters, not just what it contains")
            elif element == "mentions_outcomes":
                suggestions.append("Include specific business outcomes and impacts")
            elif element == "actionable_insights":
                suggestions.append("Provide actionable insights for optimization and improvement")
        
        # Moving services context improvements
        moving_context = validation_results.get("moving_services_context", {})
        if moving_context.get("assessment") in ["partial", "missing"]:
            suggestions.append("Demonstrate deeper understanding of moving services industry operations")
        
        # Actionability improvements
        actionability = validation_results.get("actionability_assessment", {})
        if actionability.get("business_questions_count", 0) == 0:
            suggestions.append("Add specific business questions this data can answer")
        
        return suggestions