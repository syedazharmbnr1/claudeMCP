#!/usr/bin/env python3
import os
import json
import logging
import duckdb
import threading
import time
import asyncio
import psutil
from datetime import datetime
from collections.abc import Sequence
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from logging.handlers import RotatingFileHandler

from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# DuckDB cache management
duckdb_cache = {}
cache_access_times = {}

# Constants and Environment Variables
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
DB_PATH = os.path.join(os.path.dirname(__file__), "db/persistent_database_claude.db")
MAX_MEMORY = os.getenv('DUCKDB_MAX_MEMORY', 'AUTO')
CPU_THREADS = os.getenv('DUCKDB_THREADS', 'AUTO')
QUERY_TIMEOUT = int(os.getenv('DUCKDB_QUERY_TIMEOUT', '360'))
ENABLE_OBJECT_CACHE = os.getenv('DUCKDB_ENABLE_OBJECT_CACHE', 'true').lower() == 'true'
ENABLE_HTTP_METADATA = os.getenv('DUCKDB_ENABLE_HTTP_METADATA', 'true').lower() == 'true'
PRESERVE_INSERTION_ORDER = os.getenv('DUCKDB_PRESERVE_INSERTION_ORDER', 'false').lower() == 'true'
TEMP_DIRECTORY = os.getenv('DUCKDB_TEMP_DIRECTORY', '/tmp/duckdb')

class QueryRequest(BaseModel):
    csv_file_path: str
    query: str

def setup_logging():
    """Configure logging with rotation and formatting"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Create handlers for different log types
    log_file = os.path.join(LOG_DIR, 'duckdb_server.log')
    query_log_file = os.path.join(LOG_DIR, 'queries.log')
    
    # Main logger setup
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(process)d:%(thread)d]'
    )
    
    # Main log handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Query log handler with rotation
    query_handler = RotatingFileHandler(
        query_log_file, maxBytes=10*1024*1024, backupCount=5
    )
    query_handler.setFormatter(formatter)
    
    # Setup main logger
    logger = logging.getLogger("fastapi-mcp-server")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Setup query logger
    query_logger = logging.getLogger("query-logger")
    query_logger.setLevel(logging.INFO)
    query_logger.addHandler(query_handler)
    
    return logger, query_logger

class MCPFastAPIServer:
    def __init__(self):
        # Create necessary directories
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        os.makedirs(TEMP_DIRECTORY, exist_ok=True)
        
        self.app = Server("fastapi-mcp-server")
        self.fastapi_app = FastAPI()
        
        # Setup logging
        self.logger, self.query_logger = setup_logging()
        
        # Add CORS middleware
        self.fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Set up handlers
        self.setup_handlers()
        self.setup_fastapi_routes()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self.cleanup_duckdb_connections, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info(f"Server initialized with threads={CPU_THREADS} and max_memory={MAX_MEMORY}")

    def get_duckdb_config(self):
        """Get DuckDB configuration from environment variables"""
        return {
            'max_memory': MAX_MEMORY,
            'threads': CPU_THREADS,
            'max_query_execution_time': QUERY_TIMEOUT,
            'enable_http_metadata': str(ENABLE_HTTP_METADATA).lower(),
            'temp_directory': TEMP_DIRECTORY,
            'preserve_insertion_order': str(PRESERVE_INSERTION_ORDER).lower(),
            'enable_object_cache': str(ENABLE_OBJECT_CACHE).lower()
        }

    def load_csv_into_duckdb(self, csv_file_path: str) -> duckdb.DuckDBPyConnection:
        """Load CSV into DuckDB with optimized configuration"""
        if not self.is_valid_csv_path(csv_file_path):
            raise ValueError(f"Invalid or non-existent CSV file path: {csv_file_path}")

        cache_key = self.get_cache_key(csv_file_path)

        if cache_key in duckdb_cache:
            cache_access_times[cache_key] = time.time()
            return duckdb_cache[cache_key]
        
        try:
            # Create connection with basic config
            conn = duckdb.connect(database=DB_PATH)
            
            # Set basic pragmas
            if MAX_MEMORY != 'AUTO':
                conn.execute(f"PRAGMA memory_limit='{MAX_MEMORY}';")
            if CPU_THREADS != 'AUTO':
                conn.execute(f"PRAGMA threads={CPU_THREADS};")
            
            # Create table with optimized settings
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS data AS 
                SELECT * FROM read_csv_auto(
                    '{csv_file_path}',
                    sample_size=1000,
                    all_varchar=0
                );
            """)
            
            duckdb_cache[cache_key] = conn
            cache_access_times[cache_key] = time.time()
            self.logger.info(f"Successfully loaded CSV: {csv_file_path}")
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {str(e)}", exc_info=True)
            raise

    async def log_request(self, request: Request, query: str):
        """Log detailed request information"""
        client_host = request.client.host if request.client else "Unknown"
        headers = dict(request.headers)
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_host,
            "query": query,
            "headers": headers,
            "method": request.method,
            "path": request.url.path
        }
        self.query_logger.info(f"Request: {json.dumps(request_info, indent=2)}")

    async def execute_query_internal(self, csv_file_path: str, query: str, request: Optional[Request] = None):
        """Execute DuckDB query with logging and monitoring"""
        start_time = time.time()
        try:
            if request:
                await self.log_request(request, query)
            
            conn = self.load_csv_into_duckdb(csv_file_path)
            result = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.description]
            processed_data = [dict(zip(columns, row)) for row in result]
            
            execution_time = time.time() - start_time
            self.query_logger.info(f"Query executed successfully in {execution_time:.2f} seconds")
            
            return {
                "success": True,
                "data": {
                    "columns": columns,
                    "rows": processed_data,
                    "rowCount": len(processed_data)
                },
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Query failed after {execution_time:.2f} seconds: {str(e)}", 
                exc_info=True
            )
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }

    def setup_fastapi_routes(self):
        @self.fastapi_app.post("/execute_query")
        async def execute_query(request: Request, query_request: QueryRequest):
            try:
                result = await self.execute_query_internal(
                    csv_file_path=query_request.csv_file_path,
                    query=query_request.query,
                    request=request
                )
                return result
            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.fastapi_app.get("/health")
        async def health_check():
            return {"status": "healthy", "server": "fastapi-mcp-server"}

    def is_valid_csv_path(self, csv_file_path: str) -> bool:
        """Validate if the CSV path is safe to use"""
        csv_file_path = os.path.abspath(csv_file_path)
        return csv_file_path.endswith('.csv') and os.path.exists(csv_file_path)

    def get_cache_key(self, csv_path: str) -> str:
        """Generate cache key based on file path and modification time"""
        mod_time = os.path.getmtime(csv_path)
        return f"{csv_path}:{mod_time}"

    def cleanup_duckdb_connections(self):
        """Cleanup unused DuckDB connections periodically"""
        while True:
            time.sleep(300)  # Check every 5 minutes
            current_time = time.time()
            to_delete = []
            
            for cache_key, last_access in cache_access_times.items():
                if current_time - last_access > 600:  # 10 minutes timeout
                    conn = duckdb_cache.get(cache_key)
                    if conn:
                        conn.close()
                    to_delete.append(cache_key)
                    
            for cache_key in to_delete:
                del duckdb_cache[cache_key]
                del cache_access_times[cache_key]
            
            self.logger.info(f"Cleaned up {len(to_delete)} unused connections")

    def setup_handlers(self):
        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="execute_query",
                    description="Execute DuckDB query on CSV file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "csv_file_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "query": {
                                "type": "string",
                                "description": "DuckDB SQL query to execute"
                            }
                        },
                        "required": ["csv_file_path", "query"]
                    }
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            if name != "execute_query":
                raise ValueError(f"Unknown tool: {name}")

            try:
                result = await self.execute_query_internal(
                    csv_file_path=arguments.get("csv_file_path"),
                    query=arguments.get("query")
                )

                # Convert datetime objects to strings for JSON serialization
                if result["data"]["rows"]:
                    for row in result["data"]["rows"]:
                        for key, value in row.items():
                            if isinstance(value, (datetime.date, datetime.datetime)):
                                row[key] = value.isoformat()
                
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )
                ]

            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": str(e)
                        }, indent=2)
                    )
                ]

    async def run_fastapi(self):
        """Run the FastAPI server"""
        config = uvicorn.Config(
            self.fastapi_app, 
            host="0.0.0.0", 
            port=8010, 
            log_level="info",
            reload=False
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_mcp(self):
        """Run the MCP server"""
        from mcp.server.stdio import stdio_server
        self.logger.info("Starting MCP Server")
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                self.app.create_initialization_options()
            )

    async def run(self):
        """Main entry point for the server"""
        self.logger.info("Starting FastAPI MCP Server")
        try:
            await asyncio.gather(
                self.run_fastapi(),
                self.run_mcp()
            )
        except Exception as e:
            self.logger.error(f"Server error: {str(e)}", exc_info=True)
            raise

async def main():
    try:
        server = MCPFastAPIServer()
        await server.run()
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)