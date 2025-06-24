# working_crewai_test.py - Test CrewAI with correct API for version 0.126.0

import os
import sys
from dotenv import load_dotenv

def test_working_crewai():
    """Test CrewAI with the correct API for version 0.126.0"""
    
    print("🚀 WORKING CREWAI TEST (Version 0.126.0)")
    print("=" * 50)
    
    load_dotenv()
    
    try:
        from crewai import Agent, LLM, Crew, Task, Process
        
        model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE')
        print(f"🎯 Using model: {model_id}")
        
        # Create LLM - this works based on your diagnostic
        llm = LLM(model=f"bedrock/{model_id}", temperature=0.1)
        print(f"✅ LLM created successfully")
        
        # Create agent
        agent = Agent(
            role="Test Agent",
            goal="Respond with simple messages",
            backstory="A simple test agent for verification",
            llm=llm,
            verbose=False  # Reduce noise
        )
        print(f"✅ Agent created successfully")
        
        # Create task
        task = Task(
            description="Say exactly: 'CrewAI is working!'",
            expected_output="The exact phrase 'CrewAI is working!'",
            agent=agent
        )
        print(f"✅ Task created successfully")
        
        # Create crew (this is the correct way for 0.126.0)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False  # Reduce output for cleaner test
        )
        print(f"✅ Crew created successfully")
        
        # Execute crew (NOT task.execute())
        print(f"🚀 Running crew...")
        result = crew.kickoff()
        
        print(f"✅ SUCCESS! CrewAI is working!")
        print(f"📝 Result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ CrewAI test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def test_simple_tools():
    """Test if tools work with the working CrewAI"""
    
    print(f"\n🛠️  TESTING TOOLS WITH WORKING CREWAI")
    print("=" * 40)
    
    try:
        from crewai import Agent, LLM, Crew, Task, Process
        from tools.adaptive_weaviate_tools import adaptive_business_context_analyzer
        
        model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE')
        
        # Create LLM
        llm = LLM(model=f"bedrock/{model_id}", temperature=0.1)
        
        # Create agent with tools
        agent = Agent(
            role="Tool Test Agent",
            goal="Test tool functionality",
            backstory="An agent that tests tools",
            llm=llm,
            tools=[adaptive_business_context_analyzer],
            verbose=False
        )
        
        # Create task that uses tools
        task = Task(
            description="Use the adaptive_business_context_analyzer tool with the query 'test customer email'",
            expected_output="Output from the adaptive_business_context_analyzer tool",
            agent=agent
        )
        
        # Create and run crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        print(f"🚀 Running crew with tools...")
        result = crew.kickoff()
        
        print(f"✅ Tools working with CrewAI!")
        print(f"📝 Result preview: {str(result)[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return False

def main():
    """Run working CrewAI tests"""
    
    print("🔧 CREWAI COMPATIBILITY TEST & FIX")
    print("=" * 50)
    
    # Test 1: Basic CrewAI functionality
    basic_works = test_working_crewai()
    
    if basic_works:
        print(f"\n🎉 BASIC CREWAI WORKS!")
        
        # Test 2: Tools integration
        tools_work = test_simple_tools()
        
        if tools_work:
            print(f"\n🎉 TOOLS INTEGRATION WORKS!")
            print(f"\n✅ YOUR PIPELINE SHOULD WORK NOW!")
            print(f"🚀 Try running: python run.py")
        else:
            print(f"\n⚠️  Basic CrewAI works but tools have issues")
            print(f"Check your adaptive_weaviate_tools.py")
    else:
        print(f"\n❌ BASIC CREWAI STILL HAS ISSUES")
        print(f"💡 Try upgrading CrewAI:")
        print(f"   pip install crewai[tools] --upgrade")

if __name__ == "__main__":
    main()