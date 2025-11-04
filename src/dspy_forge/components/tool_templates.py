"""
DSPy tool component templates.

Handles MCP Tool and UC Function tool types.
"""
import os
import asyncio

from typing import Dict, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

from dspy.utils.mcp import _convert_mcp_tool_result
from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

from dspy_forge.core.templates import NodeTemplate, CodeGenerationContext
from dspy_forge.core.logging import get_logger

logger = get_logger(__name__)


class BaseToolTemplate(NodeTemplate):
    """Base template for tool nodes"""

    def initialize(self, context: Any):
        """
        Tools don't have a direct execution component.
        They are collected by ReAct nodes that connect to them.
        Returns None as tools are not directly executable.
        """
        return None

    def generate_code(self, context: CodeGenerationContext) -> Dict[str, Any]:
        """Generate code for tool node - handled by ReAct template"""
        # Tool nodes don't generate standalone code
        # They are collected and used by ReAct nodes
        return {
            'signature': '',
            'instance': '',
            'forward': '',
            'dependencies': [],
            'instance_var': None
        }

    def generate_tool_loading_method(self, var_prefix: str) -> str:
        """Generate method to load tools - override in subclasses"""
        raise NotImplementedError("Subclasses must implement generate_tool_loading_method")


class MCPToolTemplate(BaseToolTemplate):
    """Template for MCP Tool nodes"""

    def get_tool_config(self) -> Dict[str, Any]:
        """Extract MCP tool configuration"""
        # Support both snake_case (backend) and camelCase (frontend)
        return {
            'tool_type': 'mcp',
            'tool_name': self.node_data.get('tool_name') or self.node_data.get('toolName', 'Unnamed Tool'),
            'description': self.node_data.get('description', ''),
            'mcp_url': self.node_data.get('mcp_url') or self.node_data.get('mcpUrl', ''),
            'mcp_headers': self.node_data.get('mcp_headers') or self.node_data.get('mcpHeaders', []),
        }
    
    @staticmethod
    def convert_mcp_tool(client_info: Dict[str, Any], tool: "mcp.types.Tool") -> Tool:
        """Convert an MCP tool to DSPy Tool format"""
        args, arg_types, arg_desc = convert_input_schema_to_tool_args(tool.inputSchema)

        # Convert the MCP tool and Session to a single async method
        async def func(*args, **kwargs):
            async with streamablehttp_client(
                url=client_info["url"],
                headers=client_info["headers"]
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool.name, arguments=kwargs)
                    return _convert_mcp_tool_result(result)

        return Tool(
            func=func,
            name=tool.name,
            desc=tool.description,
            args=args,
            arg_types=arg_types,
            arg_desc=arg_desc
        )

    async def list_mcp_tools(self) -> list:
        """Load MCP tools asynchronously - called during execution service setup"""
        # Get configuration
        mcp_url = self.node_data.get('mcp_url') or self.node_data.get('mcpUrl', '')
        mcp_headers = self.node_data.get('mcp_headers') or self.node_data.get('mcpHeaders', [])

        if not mcp_url:
            logger.error("MCP tool missing URL")
            return []

        # Build headers dictionary
        headers = {}
        for header in mcp_headers:
            key = header.get('key', '')
            value = header.get('value', '')
            # Support both naming conventions for headers
            is_secret = header.get('is_secret') or header.get('isSecret', False)
            env_var_name = header.get('env_var_name') or header.get('envVarName', '')

            if is_secret and env_var_name:
                # Use environment variable
                env_value = os.getenv(env_var_name)
                if env_value:
                    headers[key] = env_value
                else:
                    logger.warning(f"Environment variable {env_var_name} not found for header {key}")
            else:
                # Use literal value
                headers[key] = value

        tools = []
        try:
            async with streamablehttp_client(
                url=mcp_url,
                headers=headers
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_list = await session.list_tools()

                    # Convert MCP tools to DSPy tools
                    tools = [
                        MCPToolTemplate.convert_mcp_tool(
                            {"url": mcp_url, "headers": headers}, tool
                        ) for tool in tool_list.tools
                    ]
            logger.info(f"Loaded {len(tools)} tools from MCP server: {mcp_url}")
        except Exception as e:
            logger.error(f"Failed to load MCP tools from {mcp_url}: {e}", exc_info=True)
            return []
        
        return tools

    def load_tools(self, context=None) -> list:
        """
        Load MCP tools - retrieves from context if pre-loaded, otherwise returns empty.
        MCP tools should be pre-loaded asynchronously by the execution service.
        """
        if context and hasattr(context, 'get_loaded_tools'):
            # Get pre-loaded tools from context
            return context.get_loaded_tools(self.node.id)

        logger.warning(f"No pre-loaded MCP tools found in context for node {self.node.id}")
        return []

    def generate_tool_loading_method(self, var_prefix: str) -> str:
        """Generate code for MCP client initialization"""
        tool_config = self.get_tool_config()
        mcp_url = tool_config['mcp_url']
        headers = tool_config['mcp_headers']

        # Build headers dictionary code
        header_lines = []
        header_lines.append("        headers = {")

        for header in headers:
            key = header.get('key', '')
            value = header.get('value', '')
            # Support both naming conventions
            is_secret = header.get('is_secret') or header.get('isSecret', False)
            env_var_name = header.get('env_var_name') or header.get('envVarName', '')

            if is_secret and env_var_name:
                # Use environment variable
                header_lines.append(f'            "{key}": os.getenv("{env_var_name}"),')
            else:
                # Use literal value
                header_lines.append(f'            "{key}": "{value}",')

        header_lines.append("        }")
        headers_code = '\n'.join(header_lines)

        # Generate async MCP client code that returns tools
        code = f'''
    async def _load_{var_prefix}_tools(self):
        """Load tools from MCP server: {mcp_url}"""
        import os
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

{headers_code}

        tools = []
        try:
            async with streamablehttp_client(
                url="{mcp_url}",
                headers=headers
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tool_list = await session.list_tools()
                    tools = [
                        dspy.Tool.from_mcp_tool(session, tool)
                        for tool in tool_list.tools
                    ]
                    logger.info(f"Loaded {{len(tools)}} tools from MCP server: {mcp_url}")
        except Exception as e:
            logger.error(f"Failed to load MCP tools from {mcp_url}: {{e}}")

        return tools
'''
        return code


class UCFunctionTemplate(BaseToolTemplate):
    """Template for Unity Catalog Function tool nodes"""

    def get_tool_config(self) -> Dict[str, Any]:
        """Extract UC function configuration"""
        # Support both snake_case (backend) and camelCase (frontend)
        return {
            'tool_type': 'uc_function',
            'tool_name': self.node_data.get('tool_name') or self.node_data.get('toolName', 'Unnamed Function'),
            'description': self.node_data.get('description', ''),
            'catalog': self.node_data.get('catalog', ''),
            'schema': self.node_data.get('schema', ''),
            'function_name': self.node_data.get('function_name') or self.node_data.get('functionName', ''),
        }

    def load_tools(self, context=None) -> list:
        """Load Unity Catalog function tools synchronously"""
        from dspy.adapters.types.tool import Tool as DSPyTool

        # Get configuration
        catalog = self.node_data.get('catalog', '')
        schema = self.node_data.get('schema', '')
        function_name = self.node_data.get('function_name') or self.node_data.get('functionName', '')
        description = self.node_data.get('description', '')
        tool_name = self.node_data.get('tool_name') or self.node_data.get('toolName', '')

        full_func_name = f"{catalog}.{schema}.{function_name}"

        if not all([catalog, schema, function_name]):
            logger.error(f"UC Function tool missing required fields: catalog={catalog}, schema={schema}, function_name={function_name}")
            return []

        try:
            # Initialize UC function client
            client = DatabricksFunctionClient()
            set_uc_function_client(client)

            # Get function metadata to create proper tool signature
            # The client provides function information including parameters
            
            # Create DSPy tool from UC function
            def uc_function_wrapper(**kwargs):
                """Wrapper to execute UC function"""
                try:
                    result = client.execute_function(full_func_name, kwargs)
                    return result
                except Exception as e:
                    logger.error(f"Error executing UC function {full_func_name}: {e}", exc_info=True)
                    raise

            # Create tool with proper metadata
            tool = DSPyTool(
                func=uc_function_wrapper,
                name=tool_name or function_name,
                desc=description or f"Unity Catalog function: {full_func_name}",
                args={},  # Will be populated from function metadata
                arg_types={},
                arg_desc={}
            )
            tools = [tool]
            logger.info(f"Loaded UC function: {full_func_name}")
            return tools

        except Exception as e:
            logger.error(f"Failed to load UC function {full_func_name}: {e}", exc_info=True)
            return []

    def generate_tool_loading_method(self, var_prefix: str) -> str:
        """Generate code for UC function client initialization"""
        tool_config = self.get_tool_config()
        catalog = tool_config['catalog']
        schema = tool_config['schema']
        function_name = tool_config['function_name']
        tool_name = tool_config['tool_name']
        full_func_name = f"{catalog}.{schema}.{function_name}"

        # Generate UC function loading code
        code = f'''
    def _load_{var_prefix}_tools(self):
        """Load Unity Catalog function: {full_func_name}"""
        from unitycatalog.ai.core.base import set_uc_function_client
        from unitycatalog.ai.core.databricks import DatabricksFunctionClient
        from dspy.adapters.types.tool import Tool

        tools = []
        try:
            client = DatabricksFunctionClient()
            set_uc_function_client(client)

            # Create wrapper function for UC function execution
            def uc_function_wrapper(**kwargs):
                try:
                    result = client.execute_function("{full_func_name}", kwargs)
                    return result
                except Exception as e:
                    logger.error(f"Error executing UC function {full_func_name}: {{e}}")
                    raise

            # Create DSPy tool from UC function
            tool = Tool(
                func=uc_function_wrapper,
                name="{tool_name or function_name}",
                desc="{tool_config['description'] or f'Unity Catalog function: {full_func_name}'}",
                args={{}},
                arg_types={{}},
                arg_desc={{}}
            )
            tools.append(tool)
            logger.info(f"Loaded UC function: {full_func_name}")
        except Exception as e:
            logger.error(f"Failed to load UC function {full_func_name}: {{e}}")

        return tools
'''
        return code
