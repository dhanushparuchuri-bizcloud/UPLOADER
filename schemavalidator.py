# column_discovery_debugger.py - Debug why column discovery isn't finding results

import os
import sys
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def debug_column_discovery():
    """Debug why the column discovery query isn't finding results"""
    
    try:
        from tools.adaptive_weaviate_tools import WeaviateClientSingleton
        import weaviate.classes.query as wq
        from weaviate.classes.query import Filter
        
        client = WeaviateClientSingleton.get_instance()
        if not client:
            print("❌ Cannot connect to Weaviate")
            return False
            
        print("🔍 COLUMN DISCOVERY DEBUGGING")
        print("=" * 50)
        
        column_collection = client.collections.get("Column")
        
        # Step 1: Check all available columns
        print("1️⃣ Checking all available columns...")
        all_response = column_collection.query.fetch_objects(limit=10)
        
        if not all_response.objects:
            print("❌ No columns found at all in the Column collection!")
            return False
            
        print(f"✅ Found {len(all_response.objects)} columns total")
        
        # Inspect parentAthenaTableName values
        table_names = set()
        for obj in all_response.objects:
            props = obj.properties
            table_name = props.get('parentAthenaTableName', 'NO_TABLE_NAME')
            table_names.add(table_name)
            print(f"   • Column: {props.get('columnName', 'NO_NAME')} → Table: {table_name}")
        
        print(f"\n📊 Unique parentAthenaTableName values found:")
        for table_name in sorted(table_names):
            print(f"   • '{table_name}'")
        
        # Step 2: Test specific filter that was failing
        print(f"\n2️⃣ Testing filter: parentAthenaTableName = 'amspoc3test.customer'")
        
        customer_filter = Filter.by_property("parentAthenaTableName").equal("amspoc3test.customer")
        customer_response = column_collection.query.fetch_objects(
            filters=customer_filter,
            limit=10
        )
        
        if customer_response.objects:
            print(f"✅ Found {len(customer_response.objects)} columns for 'amspoc3test.customer':")
            for obj in customer_response.objects:
                props = obj.properties
                print(f"   • {props.get('columnName', 'NO_NAME')} ({props.get('athenaDataType', 'NO_TYPE')})")
        else:
            print("❌ No columns found for 'amspoc3test.customer'")
            print("💡 This explains why the discovery query failed!")
            
            # Check if 'customer' table exists with different naming
            customer_variants = ['customer', 'amspoc3test.customer', 'Customer', 'CUSTOMER']
            print(f"\n🔍 Checking for customer table variants...")
            
            for variant in customer_variants:
                variant_filter = Filter.by_property("parentAthenaTableName").equal(variant)
                variant_response = column_collection.query.fetch_objects(
                    filters=variant_filter,
                    limit=5
                )
                
                if variant_response.objects:
                    print(f"✅ Found {len(variant_response.objects)} columns for '{variant}':")
                    for obj in variant_response.objects:
                        props = obj.properties
                        print(f"   • {props.get('columnName', 'NO_NAME')} ({props.get('athenaDataType', 'NO_TYPE')})")
                    break
            else:
                print("❌ No customer table found with any common variant")
        
        # Step 3: Test semantic search without filters
        print(f"\n3️⃣ Testing semantic search without filters...")
        semantic_response = column_collection.query.near_text(
            query="email address contact information",
            limit=5,
            return_metadata=wq.MetadataQuery(distance=True)
        )
        
        if semantic_response.objects:
            print(f"✅ Found {len(semantic_response.objects)} columns via semantic search:")
            for obj in semantic_response.objects:
                props = obj.properties
                distance = obj.metadata.distance
                table_name = props.get('parentAthenaTableName', 'NO_TABLE')
                print(f"   • {props.get('columnName', 'NO_NAME')} from {table_name} (distance: {distance:.3f})")
        else:
            print("❌ No columns found via semantic search either!")
        
        # Step 4: Check DatasetMetadata for available tables
        print(f"\n4️⃣ Checking DatasetMetadata for available tables...")
        try:
            dataset_collection = client.collections.get("DatasetMetadata")
            dataset_response = dataset_collection.query.fetch_objects(limit=10)
            
            if dataset_response.objects:
                print(f"✅ Found {len(dataset_response.objects)} datasets:")
                for obj in dataset_response.objects:
                    props = obj.properties
                    table_name = props.get('tableName', 'NO_NAME')
                    athena_name = props.get('athenaTableName', 'NO_ATHENA_NAME')
                    print(f"   • Table: {table_name} → Athena: {athena_name}")
            else:
                print("❌ No datasets found in DatasetMetadata!")
                
        except Exception as e:
            print(f"❌ Error checking DatasetMetadata: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Column discovery debugging failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fixes():
    """Suggest fixes based on debugging results"""
    
    print("\n" + "=" * 60)
    print("💡 DEBUGGING ANALYSIS & SUGGESTED FIXES")
    print("=" * 60)
    
    print("""
🎯 LIKELY ISSUES & FIXES:

1. **Table Name Mismatch**:
   - The filter looks for 'amspoc3test.customer' 
   - But your data might use 'customer', 'Customer', or different schema prefix
   - FIX: Update the filter to use the actual table name from your data

2. **Schema Prefix Issue**:
   - Your code assumes 'amspoc3test.' prefix
   - But actual data might not have this prefix
   - FIX: Check DatasetMetadata for actual athenaTableName values

3. **Data Population Issue**:
   - Columns might not have parentAthenaTableName populated correctly
   - FIX: Verify your data ingestion process sets this field

4. **Case Sensitivity**:
   - Filter might be case-sensitive
   - FIX: Try different case variations

🔧 QUICK FIXES TO TRY:

1. **Update Fallback Schema** (adaptive_weaviate_tools.py):
   Use the actual table names found in DatasetMetadata

2. **Dynamic Table Name Discovery**:
   Instead of hardcoding 'amspoc3test.customer', discover available tables first

3. **Flexible Filtering**:
   Try multiple table name variants in the filter

🚀 **IMMEDIATE ACTION**:
   Run: python run.py 
   
   The fallback schema should now work even if Weaviate discovery fails,
   and your execution engine will self-correct any remaining column name issues.
""")

def create_dynamic_table_discovery():
    """Create a function to dynamically discover available tables"""
    
    print("\n" + "=" * 50)
    print("🔧 DYNAMIC TABLE DISCOVERY SOLUTION")
    print("=" * 50)
    
    solution_code = '''
def discover_available_tables(client):
    """Dynamically discover what tables are actually available"""
    
    try:
        # Get all datasets
        dataset_collection = client.collections.get("DatasetMetadata")
        dataset_response = dataset_collection.query.fetch_objects(limit=20)
        
        available_tables = {}
        for obj in dataset_response.objects:
            props = obj.properties
            table_name = props.get('tableName', '')
            athena_name = props.get('athenaTableName', '')
            
            if table_name and athena_name:
                available_tables[table_name] = athena_name
                
        return available_tables
        
    except Exception as e:
        print(f"Table discovery failed: {e}")
        return {}

# Usage in adaptive pipeline:
available_tables = discover_available_tables(client)
print(f"Available tables: {available_tables}")

# Use actual table names instead of hardcoded ones:
if 'customer' in available_tables:
    customer_athena_name = available_tables['customer']
    customer_filter = Filter.by_property("parentAthenaTableName").equal(customer_athena_name)
'''
    
    print("💡 Add this dynamic discovery to your adaptive tools:")
    print(solution_code)

if __name__ == "__main__":
    print("🔍 COLUMN DISCOVERY DEBUGGER")
    print("This tool debugs why column discovery queries aren't finding results")
    print("=" * 80)
    
    success = debug_column_discovery()
    
    if success:
        suggest_fixes()
        create_dynamic_table_discovery()
        
        print("\n🚀 NEXT STEPS:")
        print("1. Review the debugging output above")
        print("2. Note the actual table names found in your data")
        print("3. Try running: python run.py")
        print("4. The fallback schema + execution engine should handle any remaining issues")
    else:
        print("\n❌ Debugging failed - check your Weaviate connection and data")