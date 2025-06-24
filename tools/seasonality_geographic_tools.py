"""
Seasonality and Geographic analysis tools for moving services
"""

import json
import logging
import boto3
import os
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from .base_discovery_tools import SchemaDiscoveryMixin

logger = logging.getLogger(__name__)

class SeasonalityDetectionTool(BaseTool, SchemaDiscoveryMixin):
    name: str = "Comprehensive Seasonality Pattern Detector"
    description: str = "Detect multiple types of seasonal patterns affecting moving services business"
    
    # Declare Pydantic fields for the boto3 clients and configuration
    athena_client: Any = None
    s3_results_location: str = ""
    
    def __init__(self):
        # Initialize the base tool first
        super().__init__()
        # Then set up the AWS clients
        self.athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.s3_results_location = os.getenv('ATHENA_RESULTS_BUCKET', 's3://your-athena-results-bucket/')
    
    def _run(self, table_name: str, date_column: str = None) -> str:
        """Detect comprehensive seasonal patterns in moving services data"""
        try:
            logger.info(f"🌱 Analyzing seasonality patterns in {table_name}")
            
            # Discover schema
            schema = self.discover_table_schema(table_name)
            
            # Find date columns if not specified
            if not date_column:
                date_columns = self._find_date_columns(schema)
                if not date_columns:
                    return json.dumps({
                        "error": "No date columns found for seasonality analysis",
                        "available_columns": list(schema.keys())
                    })
                date_column = date_columns[0]  # Use first date column
            
            if date_column not in schema:
                return json.dumps({
                    "error": f"Date column {date_column} not found",
                    "available_columns": list(schema.keys())
                })
            
            # Comprehensive seasonality analysis
            seasonality_analysis = {
                "temporal_patterns": self._analyze_temporal_patterns(table_name, date_column, schema),
                "moving_industry_seasonality": self._analyze_moving_seasonality(table_name, date_column, schema),
                "business_cycle_patterns": self._analyze_business_cycles(table_name, date_column, schema),
                "external_factor_indicators": self._analyze_external_factors(table_name, date_column, schema)
            }
            
            # Generate insights
            insights = self._generate_seasonality_insights(seasonality_analysis, schema)
            
            return json.dumps({
                "table_name": table_name,
                "date_column_analyzed": date_column,
                "seasonality_analysis": seasonality_analysis,
                "business_insights": insights,
                "recommendation_summary": self._generate_seasonality_recommendations(seasonality_analysis)
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Seasonality detection failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _find_date_columns(self, schema: Dict) -> List[str]:
        """Find columns that appear to contain date/time data"""
        date_columns = []
        for col_name, col_info in schema.items():
            purpose = col_info.get("inferred_purpose", {})
            if purpose.get("category") == "temporal":
                date_columns.append(col_name)
        return date_columns
    
    def _analyze_temporal_patterns(self, table_name: str, date_column: str, schema: Dict) -> Dict:
        """Detect various temporal patterns using statistical methods"""
        patterns = {}
        
        try:
            # Monthly volume analysis
            monthly_query = f"""
            SELECT 
                EXTRACT(MONTH FROM {date_column}) as month,
                COUNT(*) as record_count,
                EXTRACT(YEAR FROM {date_column}) as year
            FROM {table_name} 
            WHERE {date_column} IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM {date_column}), EXTRACT(MONTH FROM {date_column})
            ORDER BY year, month
            """
            
            monthly_data = self.execute_athena_query(monthly_query)
            if monthly_data:
                patterns["monthly_volume"] = self._calculate_seasonality_strength(monthly_data, "record_count")
            
            # Day of week analysis
            dow_query = f"""
            SELECT 
                EXTRACT(DOW FROM {date_column}) as day_of_week,
                COUNT(*) as record_count
            FROM {table_name} 
            WHERE {date_column} IS NOT NULL
            GROUP BY EXTRACT(DOW FROM {date_column})
            ORDER BY day_of_week
            """
            
            dow_data = self.execute_athena_query(dow_query)
            if dow_data:
                patterns["day_of_week"] = self._calculate_seasonality_strength(dow_data, "record_count")
            
            # Quarterly analysis
            quarterly_query = f"""
            SELECT 
                EXTRACT(QUARTER FROM {date_column}) as quarter,
                COUNT(*) as record_count,
                EXTRACT(YEAR FROM {date_column}) as year
            FROM {table_name} 
            WHERE {date_column} IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM {date_column}), EXTRACT(QUARTER FROM {date_column})
            ORDER BY year, quarter
            """
            
            quarterly_data = self.execute_athena_query(quarterly_query)
            if quarterly_data:
                patterns["quarterly"] = self._calculate_seasonality_strength(quarterly_data, "record_count")
            
        except Exception as e:
            logger.warning(f"Temporal pattern analysis failed: {e}")
            patterns["error"] = str(e)
        
        return patterns
    
    def _analyze_moving_seasonality(self, table_name: str, date_column: str, schema: Dict) -> Dict:
        """Analyze moving industry specific seasonal factors"""
        moving_patterns = {}
        
        try:
            # Weather season impact
            weather_query = f"""
            SELECT 
                CASE 
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (12,1,2) THEN 'winter'
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (3,4,5) THEN 'spring'
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (6,7,8) THEN 'summer'
                    ELSE 'fall'
                END as weather_season,
                COUNT(*) as move_volume
            FROM {table_name}
            WHERE {date_column} IS NOT NULL
            GROUP BY weather_season
            """
            
            weather_data = self.execute_athena_query(weather_query)
            if weather_data:
                moving_patterns["weather_seasons"] = {
                    "data": weather_data,
                    "analysis": self._interpret_weather_seasonality(weather_data)
                }
            
            # School calendar correlation
            school_query = f"""
            SELECT 
                CASE 
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (6,7,8) THEN 'summer_break'
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (12,1) THEN 'winter_break'
                    WHEN EXTRACT(MONTH FROM {date_column}) = 5 THEN 'school_year_end'
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (8,9) THEN 'school_year_start'
                    ELSE 'school_year'
                END as school_period,
                COUNT(*) as move_volume
            FROM {table_name}
            WHERE {date_column} IS NOT NULL
            GROUP BY school_period
            """
            
            school_data = self.execute_athena_query(school_query)
            if school_data:
                moving_patterns["school_calendar"] = {
                    "data": school_data,
                    "analysis": self._interpret_school_seasonality(school_data)
                }
            
            # Housing market periods
            housing_query = f"""
            SELECT 
                CASE 
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (3,4,5,6) THEN 'spring_buying_season'
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (7,8,9) THEN 'summer_moving_peak'
                    WHEN EXTRACT(MONTH FROM {date_column}) IN (10,11) THEN 'fall_market'
                    ELSE 'winter_slow_period'
                END as housing_period,
                COUNT(*) as move_volume
            FROM {table_name}
            WHERE {date_column} IS NOT NULL
            GROUP BY housing_period
            """
            
            housing_data = self.execute_athena_query(housing_query)
            if housing_data:
                moving_patterns["housing_market"] = {
                    "data": housing_data,
                    "analysis": self._interpret_housing_seasonality(housing_data)
                }
            
        except Exception as e:
            logger.warning(f"Moving seasonality analysis failed: {e}")
            moving_patterns["error"] = str(e)
        
        return moving_patterns
    
    def _analyze_business_cycles(self, table_name: str, date_column: str, schema: Dict) -> Dict:
        """Analyze business cycle patterns"""
        business_cycles = {}
        
        # Find financial columns for revenue/cost seasonality
        financial_columns = [
            col for col, info in schema.items()
            if info.get("inferred_purpose", {}).get("category") == "financial"
        ]
        
        for fin_col in financial_columns[:3]:  # Limit to avoid too many queries
            try:
                financial_seasonal_query = f"""
                SELECT 
                    EXTRACT(MONTH FROM {date_column}) as month,
                    AVG({fin_col}) as avg_value,
                    COUNT(*) as sample_size
                FROM {table_name} 
                WHERE {date_column} IS NOT NULL AND {fin_col} IS NOT NULL
                GROUP BY EXTRACT(MONTH FROM {date_column})
                HAVING COUNT(*) > 5
                ORDER BY month
                """
                
                financial_data = self.execute_athena_query(financial_seasonal_query)
                if financial_data:
                    business_cycles[f"{fin_col}_seasonality"] = {
                        "data": financial_data,
                        "seasonality_strength": self._calculate_seasonality_strength(financial_data, "avg_value"),
                        "business_impact": self._interpret_financial_seasonality(fin_col, financial_data)
                    }
                    
            except Exception as e:
                logger.warning(f"Business cycle analysis failed for {fin_col}: {e}")
        
        return business_cycles
    
    def _analyze_external_factors(self, table_name: str, date_column: str, schema: Dict) -> Dict:
        """Analyze correlation with external factors"""
        # This would integrate with external data in a real implementation
        return {
            "economic_indicators": {
                "analysis": "External economic data integration would be needed",
                "potential_correlations": [
                    "Housing market trends affect moving volume",
                    "Economic recessions reduce luxury moves but increase downsizing",
                    "Interest rates affect home buying and subsequent moving"
                ]
            },
            "demographic_events": {
                "analysis": "Demographic event correlation analysis",
                "detectable_patterns": [
                    "Military PCS (Permanent Change of Station) seasons",
                    "Corporate fiscal year relocations", 
                    "College enrollment cycles"
                ]
            }
        }
    
    def _calculate_seasonality_strength(self, data: List[Dict], value_column: str) -> Dict:
        """Calculate statistical strength of seasonal patterns"""
        if not data or len(data) < 4:
            return {"strength": "insufficient_data", "coefficient_of_variation": None}
        
        try:
            values = [float(row[value_column]) for row in data if row.get(value_column) is not None]
            
            if not values:
                return {"strength": "no_data", "coefficient_of_variation": None}
            
            mean_value = sum(values) / len(values)
            variance = sum((x - mean_value) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            coefficient_of_variation = std_dev / mean_value if mean_value > 0 else 0
            
            # Interpret seasonality strength
            if coefficient_of_variation > 0.5:
                strength = "high_seasonality"
            elif coefficient_of_variation > 0.25:
                strength = "moderate_seasonality"
            elif coefficient_of_variation > 0.1:
                strength = "low_seasonality"
            else:
                strength = "no_significant_seasonality"
            
            # Find peak and trough periods
            max_entry = max(data, key=lambda x: float(x.get(value_column, 0)))
            min_entry = min(data, key=lambda x: float(x.get(value_column, 0)))
            
            return {
                "strength": strength,
                "coefficient_of_variation": coefficient_of_variation,
                "peak_period": max_entry,
                "trough_period": min_entry,
                "seasonal_factor": max(values) / min(values) if min(values) > 0 else None
            }
            
        except Exception as e:
            return {"strength": "calculation_error", "error": str(e)}
    
    def _interpret_weather_seasonality(self, weather_data: List[Dict]) -> str:
        """Interpret weather seasonality for moving services"""
        season_volumes = {row["weather_season"]: int(row["move_volume"]) for row in weather_data}
        
        max_season = max(season_volumes, key=season_volumes.get)
        min_season = min(season_volumes, key=season_volumes.get)
        
        interpretations = {
            "summer": "Peak moving season - favorable weather and school breaks",
            "spring": "Strong moving season - mild weather and housing market activity",
            "fall": "Moderate moving season - back-to-school and pre-winter relocations",
            "winter": "Slow moving season - weather challenges and holiday disruptions"
        }
        
        return f"Peak season: {max_season} ({interpretations.get(max_season, 'Unknown')}). " \
               f"Slow season: {min_season} ({interpretations.get(min_season, 'Unknown')})"
    
    def _interpret_school_seasonality(self, school_data: List[Dict]) -> str:
        """Interpret school calendar seasonality"""
        school_volumes = {row["school_period"]: int(row["move_volume"]) for row in school_data}
        
        summer_moves = school_volumes.get("summer_break", 0)
        school_year_moves = school_volumes.get("school_year", 0)
        
        if summer_moves > school_year_moves * 1.2:
            return "Strong summer break correlation - families move when school is out"
        else:
            return "Moderate school calendar impact - moves spread throughout year"
    
    def _interpret_housing_seasonality(self, housing_data: List[Dict]) -> str:
        """Interpret housing market seasonality"""
        housing_volumes = {row["housing_period"]: int(row["move_volume"]) for row in housing_data}
        
        spring_volume = housing_volumes.get("spring_buying_season", 0)
        summer_volume = housing_volumes.get("summer_moving_peak", 0)
        
        if spring_volume + summer_volume > sum(housing_volumes.values()) * 0.6:
            return "Strong correlation with housing market - peak activity in spring/summer"
        else:
            return "Moderate housing market correlation - steady activity year-round"
    
    def _interpret_financial_seasonality(self, column_name: str, financial_data: List[Dict]) -> str:
        """Interpret financial seasonality patterns"""
        seasonality = self._calculate_seasonality_strength(financial_data, "avg_value")
        strength = seasonality.get("strength", "unknown")
        
        if strength in ["high_seasonality", "moderate_seasonality"]:
            peak = seasonality.get("peak_period", {})
            trough = seasonality.get("trough_period", {})
            
            return f"Seasonal {column_name}: peaks in month {peak.get('month', 'unknown')}, " \
                   f"lowest in month {trough.get('month', 'unknown')}"
        else:
            return f"Stable {column_name} with minimal seasonal variation"
    
    def _generate_seasonality_insights(self, analysis: Dict, schema: Dict) -> Dict:
        """Generate business insights from seasonality analysis"""
        insights = {
            "moving_industry_insights": [],
            "business_planning_recommendations": [],
            "operational_implications": [],
            "financial_impact_patterns": []
        }
        
        # Analyze temporal patterns
        temporal = analysis.get("temporal_patterns", {})
        monthly = temporal.get("monthly_volume", {})
        
        if monthly.get("strength") in ["high_seasonality", "moderate_seasonality"]:
            insights["moving_industry_insights"].append(
                "Strong seasonal patterns detected - typical of moving services industry"
            )
            insights["business_planning_recommendations"].append(
                "Adjust crew scheduling and equipment allocation based on seasonal demand"
            )
        
        # Analyze moving-specific patterns
        moving_seasonality = analysis.get("moving_industry_seasonality", {})
        
        if "weather_seasons" in moving_seasonality:
            insights["operational_implications"].append(
                "Weather seasonality affects operations - plan for weather-related delays and costs"
            )
        
        if "school_calendar" in moving_seasonality:
            insights["business_planning_recommendations"].append(
                "School calendar drives family moves - optimize marketing and capacity for peak periods"
            )
        
        # Analyze financial patterns
        business_cycles = analysis.get("business_cycle_patterns", {})
        financial_seasonal_columns = [k for k in business_cycles.keys() if "_seasonality" in k]
        
        if financial_seasonal_columns:
            insights["financial_impact_patterns"].append(
                f"Financial seasonality detected in: {', '.join([k.replace('_seasonality', '') for k in financial_seasonal_columns])}"
            )
        
        return insights
    
    def _generate_seasonality_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on seasonality analysis"""
        recommendations = []
        
        temporal = analysis.get("temporal_patterns", {})
        moving_patterns = analysis.get("moving_industry_seasonality", {})
        
        # Volume seasonality recommendations
        monthly = temporal.get("monthly_volume", {})
        if monthly.get("strength") in ["high_seasonality", "moderate_seasonality"]:
            peak = monthly.get("peak_period", {})
            trough = monthly.get("trough_period", {})
            
            if peak.get("month"):
                recommendations.append(
                    f"Peak demand in month {peak['month']} - increase crew capacity and equipment availability"
                )
            
            if trough.get("month"):
                recommendations.append(
                    f"Low demand in month {trough['month']} - schedule maintenance and training during slow period"
                )
        
        # Moving industry specific recommendations
        if "weather_seasons" in moving_patterns:
            recommendations.append(
                "Implement weather contingency plans and adjust pricing for seasonal risk factors"
            )
        
        if "school_calendar" in moving_patterns:
            recommendations.append(
                "Align marketing campaigns with school calendar - target families before summer break"
            )
        
        if not recommendations:
            recommendations.append(
                "No significant seasonal patterns detected - maintain consistent operations year-round"
            )
        
        return recommendations


class GeographicVariationTool(BaseTool, SchemaDiscoveryMixin):
    name: str = "Comprehensive Geographic Variation Analyzer"
    description: str = "Analyze multi-dimensional geographic variations in moving services business"
    
    # Declare Pydantic fields for the boto3 clients and configuration
    athena_client: Any = None
    s3_results_location: str = ""
    
    def __init__(self):
        # Initialize the base tool first
        super().__init__()
        # Then set up the AWS clients
        self.athena_client = boto3.client('athena', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        self.s3_results_location = os.getenv('ATHENA_RESULTS_BUCKET', 's3://your-athena-results-bucket/')
    
    def _run(self, table_name: str, metric_column: str = None) -> str:
        """Analyze comprehensive geographic variations"""
        try:
            logger.info(f"🗺️ Analyzing geographic variations in {table_name}")
            
            # Discover schema
            schema = self.discover_table_schema(table_name)
            
            # Find geographic columns
            geographic_columns = self._find_geographic_columns(schema)
            
            if not geographic_columns:
                return json.dumps({
                    "error": "No geographic columns found for variation analysis",
                    "available_columns": list(schema.keys())
                })
            
            # Find metric columns if not specified
            if not metric_column:
                metric_columns = self._find_metric_columns(schema)
                if not metric_columns:
                    return json.dumps({
                        "warning": "No numeric columns found for metric analysis",
                        "geographic_columns": geographic_columns,
                        "analysis": "Geographic distribution only"
                    })
                metric_column = metric_columns[0]  # Use first numeric column
            
            # Comprehensive geographic analysis
            geographic_analysis = {
                "distribution_analysis": self._analyze_geographic_distribution(table_name, geographic_columns),
                "metric_variations": self._analyze_metric_variations(table_name, geographic_columns, metric_column, schema),
                "market_characteristics": self._analyze_market_characteristics(table_name, geographic_columns, schema),
                "competitive_indicators": self._analyze_competitive_indicators(table_name, geographic_columns, schema)
            }
            
            # Generate insights
            insights = self._generate_geographic_insights(geographic_analysis, schema)
            
            return json.dumps({
                "table_name": table_name,
                "geographic_columns_analyzed": geographic_columns,
                "metric_column_analyzed": metric_column,
                "geographic_analysis": geographic_analysis,
                "business_insights": insights,
                "recommendations": self._generate_geographic_recommendations(geographic_analysis)
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Geographic variation analysis failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _find_geographic_columns(self, schema: Dict) -> List[str]:
        """Find columns that contain geographic data"""
        geographic_columns = []
        for col_name, col_info in schema.items():
            purpose = col_info.get("inferred_purpose", {})
            if purpose.get("category") == "geographic":
                geographic_columns.append(col_name)
        return geographic_columns
    
    def _find_metric_columns(self, schema: Dict) -> List[str]:
        """Find numeric columns suitable for metric analysis"""
        metric_columns = []
        for col_name, col_info in schema.items():
            data_type = col_info.get("data_type", "").lower()
            purpose = col_info.get("inferred_purpose", {})
            
            if data_type in ['double', 'decimal', 'bigint', 'integer', 'float'] and \
               purpose.get("category") in ["financial", "unknown"]:
                metric_columns.append(col_name)
        return metric_columns
    
    def _analyze_geographic_distribution(self, table_name: str, geo_columns: List[str]) -> Dict:
        """Analyze geographic distribution of data"""
        distribution = {}
        
        for geo_col in geo_columns:
            try:
                dist_query = f"""
                SELECT 
                    {geo_col} as location,
                    COUNT(*) as record_count,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
                FROM {table_name}
                WHERE {geo_col} IS NOT NULL
                GROUP BY {geo_col}
                ORDER BY record_count DESC
                """
                
                results = self.execute_athena_query(dist_query)
                if results:
                    distribution[geo_col] = {
                        "data": results,
                        "analysis": self._interpret_distribution(results)
                    }
                    
            except Exception as e:
                logger.warning(f"Distribution analysis failed for {geo_col}: {e}")
        
        return distribution
    
    def _analyze_metric_variations(self, table_name: str, geo_columns: List[str], metric_column: str, schema: Dict) -> Dict:
        """Analyze how metrics vary by geography"""
        variations = {}
        
        for geo_col in geo_columns:
            try:
                variation_query = f"""
                SELECT 
                    {geo_col} as location,
                    COUNT(*) as sample_size,
                    AVG({metric_column}) as avg_value,
                    STDDEV({metric_column}) as std_dev,
                    MIN({metric_column}) as min_value,
                    MAX({metric_column}) as max_value
                FROM {table_name}
                WHERE {geo_col} IS NOT NULL AND {metric_column} IS NOT NULL
                GROUP BY {geo_col}
                HAVING COUNT(*) >= 5  # Only locations with sufficient data
                ORDER BY avg_value DESC
                """
                
                results = self.execute_athena_query(variation_query)
                if results:
                    variations[geo_col] = {
                        "data": results,
                        "analysis": self._interpret_metric_variation(results, metric_column),
                        "statistical_significance": self._assess_statistical_significance(results)
                    }
                    
            except Exception as e:
                logger.warning(f"Metric variation analysis failed for {geo_col}: {e}")
        
        return variations
    
    def _analyze_market_characteristics(self, table_name: str, geo_columns: List[str], schema: Dict) -> Dict:
        """Analyze market characteristics by geography"""
        characteristics = {}
        
        # Find categorical columns that might indicate market characteristics
        categorical_columns = [
            col for col, info in schema.items()
            if info.get("inferred_purpose", {}).get("category") in ["categorical", "status"]
        ]
        
        for geo_col in geo_columns:
            geo_characteristics = {}
            
            for cat_col in categorical_columns[:3]:  # Limit to avoid too many queries
                try:
                    market_query = f"""
                    SELECT 
                        {geo_col} as location,
                        {cat_col} as category,
                        COUNT(*) as count,
                        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY {geo_col}) as percentage
                    FROM {table_name}
                    WHERE {geo_col} IS NOT NULL AND {cat_col} IS NOT NULL
                    GROUP BY {geo_col}, {cat_col}
                    ORDER BY {geo_col}, count DESC
                    """
                    
                    results = self.execute_athena_query(market_query)
                    if results:
                        geo_characteristics[cat_col] = {
                            "data": results,
                            "dominant_categories": self._find_dominant_categories(results)
                        }
                        
                except Exception as e:
                    logger.warning(f"Market characteristics analysis failed for {cat_col}: {e}")
            
            if geo_characteristics:
                characteristics[geo_col] = geo_characteristics
        
        return characteristics
    
    def _analyze_competitive_indicators(self, table_name: str, geo_columns: List[str], schema: Dict) -> Dict:
        """Analyze competitive indicators by geography"""
        # This would analyze competition-related metrics
        # For now, return market concentration analysis
        
        indicators = {}
        
        for geo_col in geo_columns:
            try:
                # Market concentration analysis
                concentration_query = f"""
                SELECT 
                    {geo_col} as location,
                    COUNT(*) as total_records,
                    COUNT(DISTINCT {geo_col}) OVER() as total_locations
                FROM {table_name}
                WHERE {geo_col} IS NOT NULL
                GROUP BY {geo_col}
                ORDER BY total_records DESC
                """
                
                results = self.execute_athena_query(concentration_query)
                if results:
                    indicators[geo_col] = {
                        "market_concentration": self._calculate_market_concentration(results),
                        "competitive_position": self._assess_competitive_position(results)
                    }
                    
            except Exception as e:
                logger.warning(f"Competitive analysis failed for {geo_col}: {e}")
        
        return indicators
    
    def _interpret_distribution(self, distribution_data: List[Dict]) -> Dict:
        """Interpret geographic distribution data"""
        if not distribution_data:
            return {"analysis": "No distribution data available"}
        
        total_locations = len(distribution_data)
        top_location = distribution_data[0]
        
        # Calculate concentration
        top_3_percentage = sum(
            float(row["percentage"]) for row in distribution_data[:3]
        ) if len(distribution_data) >= 3 else 100
        
        return {
            "total_locations": total_locations,
            "top_location": {
                "name": top_location["location"],
                "percentage": float(top_location["percentage"])
            },
            "market_concentration": {
                "top_3_percentage": top_3_percentage,
                "concentration_level": "high" if top_3_percentage > 70 else "moderate" if top_3_percentage > 50 else "low"
            }
        }
    
    def _interpret_metric_variation(self, variation_data: List[Dict], metric_column: str) -> Dict:
        """Interpret metric variation across geography"""
        if not variation_data:
            return {"analysis": "No variation data available"}
        
        values = [float(row["avg_value"]) for row in variation_data]
        
        if not values:
            return {"analysis": "No numeric values available"}
        
        # Calculate variation statistics
        mean_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)
        variation_range = max_value - min_value
        coefficient_of_variation = (max_value - min_value) / mean_value if mean_value > 0 else 0
        
        # Find highest and lowest performing locations
        highest = max(variation_data, key=lambda x: float(x["avg_value"]))
        lowest = min(variation_data, key=lambda x: float(x["avg_value"]))
        
        return {
            "variation_strength": "high" if coefficient_of_variation > 0.3 else "moderate" if coefficient_of_variation > 0.1 else "low",
            "highest_performer": {
                "location": highest["location"],
                "value": float(highest["avg_value"])
            },
            "lowest_performer": {
                "location": lowest["location"],
                "value": float(lowest["avg_value"])
            },
            "performance_gap": {
                "absolute": variation_range,
                "relative": coefficient_of_variation
            }
        }
    
    def _assess_statistical_significance(self, variation_data: List[Dict]) -> Dict:
        """Assess statistical significance of geographic variations"""
        if len(variation_data) < 2:
            return {"significance": "insufficient_data"}
        
        # Simple analysis based on sample sizes and standard deviations
        sample_sizes = [int(row["sample_size"]) for row in variation_data]
        std_devs = [float(row.get("std_dev", 0)) for row in variation_data if row.get("std_dev")]
        
        min_sample_size = min(sample_sizes) if sample_sizes else 0
        avg_std_dev = sum(std_devs) / len(std_devs) if std_devs else 0
        
        if min_sample_size >= 30 and avg_std_dev > 0:
            significance = "likely_significant"
        elif min_sample_size >= 10:
            significance = "possibly_significant"
        else:
            significance = "insufficient_sample_size"
        
        return {
            "significance": significance,
            "min_sample_size": min_sample_size,
            "avg_std_dev": avg_std_dev
        }
    
    def _find_dominant_categories(self, category_data: List[Dict]) -> Dict:
        """Find dominant categories by location"""
        dominant = {}
        
        current_location = None
        location_categories = []
        
        for row in category_data:
            location = row["location"]
            
            if location != current_location:
                if current_location and location_categories:
                    # Find dominant category for previous location
                    top_category = max(location_categories, key=lambda x: float(x["percentage"]))
                    dominant[current_location] = {
                        "category": top_category["category"],
                        "percentage": float(top_category["percentage"])
                    }
                
                current_location = location
                location_categories = []
            
            location_categories.append(row)
        
        # Handle last location
        if current_location and location_categories:
            top_category = max(location_categories, key=lambda x: float(x["percentage"]))
            dominant[current_location] = {
                "category": top_category["category"],
                "percentage": float(top_category["percentage"])
            }
        
        return dominant
    
    def _calculate_market_concentration(self, concentration_data: List[Dict]) -> Dict:
        """Calculate market concentration metrics"""
        if not concentration_data:
            return {"concentration": "no_data"}
        
        total_records = sum(int(row["total_records"]) for row in concentration_data)
        
        # Calculate HHI (Herfindahl-Hirschman Index) approximation
        market_shares = [int(row["total_records"]) / total_records for row in concentration_data]
        hhi = sum(share ** 2 for share in market_shares) * 10000
        
        if hhi > 2500:
            concentration_level = "highly_concentrated"
        elif hhi > 1500:
            concentration_level = "moderately_concentrated"
        else:
            concentration_level = "competitive"
        
        return {
            "hhi_score": hhi,
            "concentration_level": concentration_level,
            "total_markets": len(concentration_data),
            "largest_market_share": max(market_shares) * 100
        }
    
    def _assess_competitive_position(self, concentration_data: List[Dict]) -> List[str]:
        """Assess competitive position insights"""
        insights = []
        
        if not concentration_data:
            return ["No competitive data available"]
        
        total_markets = len(concentration_data)
        
        if total_markets == 1:
            insights.append("Single market operation - no geographic diversification")
        elif total_markets < 5:
            insights.append("Limited geographic presence - opportunity for expansion")
        else:
            insights.append("Multi-market presence - geographically diversified operations")
        
        # Analyze market concentration
        records_by_market = [int(row["total_records"]) for row in concentration_data]
        max_market = max(records_by_market)
        min_market = min(records_by_market)
        
        if max_market > min_market * 3:
            insights.append("Uneven market presence - some markets significantly larger")
        else:
            insights.append("Balanced market presence across geographic areas")
        
        return insights
    
    def _generate_geographic_insights(self, analysis: Dict, schema: Dict) -> Dict:
        """Generate business insights from geographic analysis"""
        insights = {
            "market_expansion_opportunities": [],
            "operational_efficiency_insights": [],
            "competitive_positioning": [],
            "cost_structure_variations": []
        }
        
        # Analyze distribution for expansion opportunities
        distribution = analysis.get("distribution_analysis", {})
        for geo_col, dist_data in distribution.items():
            dist_analysis = dist_data.get("analysis", {})
            concentration = dist_analysis.get("market_concentration", {})
            
            if concentration.get("concentration_level") == "high":
                insights["market_expansion_opportunities"].append(
                    f"High concentration in {geo_col} - consider expanding to underserved markets"
                )
        
        # Analyze metric variations for operational insights
        variations = analysis.get("metric_variations", {})
        for geo_col, var_data in variations.items():
            var_analysis = var_data.get("analysis", {})
            
            if var_analysis.get("variation_strength") == "high":
                highest = var_analysis.get("highest_performer", {})
                lowest = var_analysis.get("lowest_performer", {})
                
                insights["operational_efficiency_insights"].append(
                    f"High performance variation in {geo_col}: {highest.get('location')} performs best, "
                    f"{lowest.get('location')} needs improvement"
                )
        
        return insights
    
    def _generate_geographic_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on geographic analysis"""
        recommendations = []
        
        # Distribution-based recommendations
        distribution = analysis.get("distribution_analysis", {})
        for geo_col, dist_data in distribution.items():
            dist_analysis = dist_data.get("analysis", {})
            concentration = dist_analysis.get("market_concentration", {})
            
            if concentration.get("concentration_level") == "high":
                recommendations.append(
                    f"Consider geographic diversification - current operations highly concentrated in {geo_col}"
                )
        
        # Performance variation recommendations
        variations = analysis.get("metric_variations", {})
        for geo_col, var_data in variations.items():
            var_analysis = var_data.get("analysis", {})
            
            if var_analysis.get("variation_strength") in ["high", "moderate"]:
                highest = var_analysis.get("highest_performer", {})
                recommendations.append(
                    f"Study best practices from {highest.get('location')} to improve performance in other {geo_col} markets"
                )
        
        if not recommendations:
            recommendations.append(
                "Geographic analysis shows consistent performance - maintain current operational standards"
            )
        
        return recommendations