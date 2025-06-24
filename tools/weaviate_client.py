"""
Weaviate client singleton for CrewAI metadata pipeline
"""

import os
import logging
from typing import Optional
import weaviate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for new timeout configuration availability
try:
    from weaviate.client import AdditionalConfig, Timeout
    HAS_NEW_TIMEOUT_CONFIG = True
except ImportError:
    HAS_NEW_TIMEOUT_CONFIG = False

class WeaviateClientSingleton:
    """Singleton pattern for Weaviate client with proper connection handling"""
    
    _instance: Optional[weaviate.WeaviateClient] = None
    _connection_attempted = False

    @classmethod
    def get_instance(cls) -> Optional[weaviate.WeaviateClient]:
        """Get Weaviate client instance with connection management"""
        if cls._instance is None and not cls._connection_attempted:
            cls._connection_attempted = True
            logger.info("🔌 Connecting to Weaviate...")
            
            try:
                # Try multiple connection methods with proper timeout configuration
                weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
                
                if "localhost" in weaviate_url or "127.0.0.1" in weaviate_url:
                    # Use connect_to_local for local connections
                    if HAS_NEW_TIMEOUT_CONFIG:
                        # New timeout configuration method
                        timeout_config = AdditionalConfig(
                            timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
                        )
                        cls._instance = weaviate.connect_to_local(
                            port=8080,
                            grpc_port=50051,
                            additional_config=timeout_config
                        )
                    else:
                        # Fallback for older versions - try without timeout first
                        try:
                            cls._instance = weaviate.connect_to_local(
                                port=8080,
                                grpc_port=50051
                            )
                        except Exception:
                            # If that fails, try even simpler connection
                            cls._instance = weaviate.connect_to_local()
                else:
                    # Use connect_to_custom for remote connections
                    host = weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
                    if HAS_NEW_TIMEOUT_CONFIG:
                        timeout_config = AdditionalConfig(
                            timeout=Timeout(init=30, query=60, insert=120)
                        )
                        cls._instance = weaviate.connect_to_custom(
                            http_host=host,
                            http_port=8080,
                            http_secure=False,
                            additional_config=timeout_config
                        )
                    else:
                        cls._instance = weaviate.connect_to_custom(
                            http_host=host,
                            http_port=8080,
                            http_secure=False
                        )
                
                if cls._instance and cls._instance.is_ready():
                    logger.info("✅ Weaviate connection successful")
                    try:
                        collections = cls._instance.collections.list_all()
                        collection_names = [col.name for col in collections]
                        logger.info(f"📊 Available collections: {collection_names}")
                        
                        if not collection_names:
                            logger.warning("⚠️ No collections found - Weaviate is empty")
                            logger.info("💡 You'll need to create schema first using schema_creator.py")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Could not list collections: {e}")
                else:
                    logger.warning("❌ Weaviate not ready, will use fallback data")
                    if cls._instance:
                        cls._instance.close()
                    cls._instance = None
                    
            except Exception as e:
                logger.warning(f"❌ Weaviate connection failed: {e}, will use fallback data")
                if cls._instance:
                    try:
                        cls._instance.close()
                    except:
                        pass
                cls._instance = None
                
        return cls._instance

    @classmethod
    def close(cls):
        """Close Weaviate client connection"""
        if cls._instance:
            try:
                cls._instance.close()
                logger.info("🔌 Weaviate connection closed")
            except Exception as e:
                logger.warning(f"Warning during Weaviate close: {e}")
            finally:
                cls._instance = None
                cls._connection_attempted = False

    @classmethod
    def reset_connection(cls):
        """Reset connection state to force reconnection"""
        cls.close()
        cls._connection_attempted = False
        return cls.get_instance()

    @classmethod
    def test_connection(cls) -> bool:
        """Test if Weaviate connection is working"""
        client = cls.get_instance()
        if not client:
            return False
        
        try:
            # Test basic connectivity
            if not client.is_ready():
                return False
            
            # Test collections access
            collections = client.collections.list_all()
            logger.info(f"✅ Connection test passed - {len(collections)} collections available")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False