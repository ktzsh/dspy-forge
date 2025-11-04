import os
import uuid
import tempfile

from datetime import datetime
from typing import Dict, Any, List, Optional

from dspy_forge.storage.factory import get_storage_backend
from dspy_forge.core.logging import get_logger
from dspy_forge.utils.workflow_utils import find_end_nodes
from dspy_forge.models.workflow import Workflow, WorkflowExecution
from dspy_forge.core.dspy_runtime import CompoundProgram
from dspy_forge.models.workflow import NodeType
from dspy_forge.components.tool_templates import MCPToolTemplate
from dspy_forge.components import registry  # This will auto-register all templates

logger = get_logger(__name__)

class ExecutionContext:
    """Context for workflow execution"""
    def __init__(self, workflow: Workflow, input_data: Dict[str, Any]):
        self.workflow = workflow
        self.input_data = input_data
        self.node_outputs: Dict[str, Dict[str, Any]] = {}
        self.execution_trace: List[Dict[str, Any]] = []
        self.models: Dict[str, Any] = {}
        self.node_counts: Dict[str, int] = {}  # Track count of each module type
        self.loaded_tools: Dict[str, List[Any]] = {}  # Store pre-loaded tools by node_id

    def set_node_output(self, node_id: str, output: Dict[str, Any]):
        """Set output for a node"""
        self.node_outputs[node_id] = output

    def get_node_output(self, node_id: str) -> Dict[str, Any]:
        """Get output from a node"""
        return self.node_outputs.get(node_id, {})

    def set_loaded_tools(self, node_id: str, tools: List[Any]):
        """Store pre-loaded tools for a tool node"""
        self.loaded_tools[node_id] = tools

    def get_loaded_tools(self, node_id: str) -> List[Any]:
        """Get pre-loaded tools for a tool node"""
        return self.loaded_tools.get(node_id, [])
        
    def add_trace_entry(self, node_id: str, node_type: str, inputs: Dict[str, Any], outputs: Dict[str, Any], execution_time: float):
        """Add entry to execution trace"""
        self.execution_trace.append({
            'node_id': node_id,
            'node_type': node_type,
            'inputs': inputs,
            'outputs': outputs,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        })


class WorkflowExecutionEngine:
    """Engine for executing DSPy workflows"""
    
    def __init__(self):
        self.active_executions: Dict[str, WorkflowExecution] = {}
    
    async def execute_workflow(self, workflow: Workflow, input_data: Dict[str, Any]) -> WorkflowExecution:
        """Execute a workflow with given input data using CompoundProgram"""
        execution_id = str(uuid.uuid4())

        execution = WorkflowExecution(
            workflow_id=workflow.id,
            input_data=input_data,
            execution_id=execution_id,
            status="pending"
        )

        self.active_executions[execution_id] = execution

        try:
            # Update status to running
            execution.status = "running"

            storage = await get_storage_backend()
            content = await storage.get_file(
                f"workflows/{workflow.id}/program.json")

            # Create execution context
            context = ExecutionContext(workflow, input_data)

            # Pre-load MCP tools asynchronously before creating the program
            await self._preload_mcp_tools(workflow, context)

            # Create and execute CompoundProgram
            program = CompoundProgram(workflow, context)

            if content:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
                try:
                    with open(temp_file, 'w') as fh:
                        fh.write(content)
                    program.load(temp_file)
                finally:
                    os.remove(temp_file)
                logger.info(f"Loaded optimized program for workflow {workflow.id}")

            # Execute the program with input data using async forward
            await program.aforward(**input_data)

            # Get final outputs from end nodes
            end_nodes = find_end_nodes(workflow)
            final_outputs = {}
            for end_node_id in end_nodes:
                final_outputs[end_node_id] = context.get_node_output(end_node_id)

            # Include execution trace and intermediate outputs in result
            execution.result = {
                'final_outputs': final_outputs,
                'execution_trace': context.execution_trace,
                'node_outputs': context.node_outputs
            }
            execution.status = "completed"

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            execution.error = str(e)
            execution.status = "failed"

        return execution
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status"""
        return self.active_executions.get(execution_id)
    
    def get_execution_trace(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get execution trace"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return []

        # For now, return empty trace - would be populated during execution
        return []

    async def _preload_mcp_tools(self, workflow: Workflow, context: ExecutionContext):
        """Pre-load MCP tools asynchronously before workflow execution"""
        # Find all MCP tool nodes
        mcp_tool_nodes = [
            node for node in workflow.nodes
            if node.type == NodeType.TOOL and
            (node.data.get('tool_type') == 'MCP_TOOL' or node.data.get('toolType') == 'MCP_TOOL')
        ]

        # Load tools for each MCP node asynchronously with error handling
        for node in mcp_tool_nodes:
            try:
                template = MCPToolTemplate(node, workflow)
                tools = await template.list_mcp_tools()
                context.set_loaded_tools(node.id, tools)
                logger.info(f"Pre-loaded {len(tools)} MCP tools for node {node.id}")
            except Exception as e:
                logger.error(f"Failed to pre-load MCP tools for node {node.id}: {e}", exc_info=True)
                context.set_loaded_tools(node.id, [])


# Global execution engine instance
execution_engine = WorkflowExecutionEngine()