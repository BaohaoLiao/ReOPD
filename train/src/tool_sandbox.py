"""
Tool sandbox module for safe code execution and tool management.

This module provides:
- PythonSandbox: Safe Python code execution environment
- ToolRegistry: Tool registration and execution management
- Memory management utilities
"""

from __future__ import annotations

import ast
import asyncio
import gc
import os
import re
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from typing import Any

import psutil

try:
    from jupyter_client import AsyncKernelManager
except ImportError:
    AsyncKernelManager = None

# Configuration for tool execution
TOOL_CONFIGS = {
    "max_turns": int(os.environ.get("TOOL_SANDBOX_MAX_TURNS", "16")),
    "max_tool_calls": int(os.environ.get("TOOL_SANDBOX_MAX_TOOL_CALLS", "16")),
    "tool_concurrency": int(os.environ.get("TOOL_SANDBOX_CONCURRENCY", "32")),
    "sandbox_backend": os.environ.get("TOOL_SANDBOX_BACKEND", "subprocess").lower(),
    # Python interpreter settings
    "python_timeout": 120,  # 2 minutes for complex calculations
    "jupyter_timeout": int(os.environ.get("TOOL_SANDBOX_JUPYTER_TIMEOUT", "300")),
    "python_memory_limit": "4GB",  # 4GB per Python process
    "python_cpu_limit": 1,
    # Memory management settings
    "max_memory_usage": 12288,  # 12GB total (75% of 16GB)
    "cleanup_threshold": 6144,  # 6GB
    "aggressive_cleanup_threshold": 3072,  # 3GB
    "force_cleanup_threshold": 9216,  # 9GB
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()


def aggressive_cleanup_memory():
    """More aggressive memory cleanup"""
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

    # Clear Python's internal caches
    import sys

    # Note: sys.intern doesn't have a clear method, so we skip this
    # Clear module cache if possible
    if hasattr(sys, "modules"):
        # Don't clear all modules, but clear some common ones that might cache data
        modules_to_clear = ["numpy", "pandas", "matplotlib", "scipy"]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "clear_cache"):
                    module.clear_cache()


def check_and_cleanup_memory():
    """Check memory usage and perform appropriate cleanup"""
    current_memory = get_memory_usage()

    if current_memory > TOOL_CONFIGS["force_cleanup_threshold"]:
        # Force aggressive cleanup
        aggressive_cleanup_memory()
        return f"Warning: High memory usage ({current_memory:.1f}MB), performed aggressive cleanup"
    elif current_memory > TOOL_CONFIGS["cleanup_threshold"]:
        # Normal cleanup
        cleanup_memory()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed cleanup"
    elif current_memory > TOOL_CONFIGS["aggressive_cleanup_threshold"]:
        # Light cleanup
        gc.collect()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed light cleanup"

    return None


class PythonSandbox:
    """Python code sandbox, provides safe code execution environment"""

    DEFAULT_IMPORT_LINES = [
        "import math",
        "import random",
        "import datetime",
        "import collections",
        "import itertools",
        "import functools",
        "import operator",
        "import statistics",
        "import decimal",
        "import fractions",
        "import sympy",
        "from sympy import *",
        "import numpy",
        "import numpy as np",
    ]

    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self._successful_code = ""
        self.allowed_modules = {
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
            "statistics",
            "decimal",
            "fractions",
            "sympy",
            "numpy"
        }

    def _build_default_import_block(self) -> str:
        """Return the bootstrap imports available to every sandbox execution."""
        return "\n".join(self.DEFAULT_IMPORT_LINES)

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check code safety by scanning for dangerous patterns"""
        # Check for dangerous operations
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"import\s+shutil",
            r"import\s+glob",
            r"import\s+pathlib",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"compile\s*\(",
            r"execfile\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"dir\s*\(",
            r"type\s*\(",
            r"isinstance\s*\(",
            r"issubclass\s*\(",
            r"super\s*\(",
            r"property\s*\(",
            r"staticmethod\s*\(",
            r"classmethod\s*\(",
            r"__\w+__",  # double underscore methods
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Code contains syntax error: {exc.msg}"

        # Check imported modules from actual Python syntax only.
        all_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    all_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                all_imports.add(node.module.split(".")[0])

        for imp in all_imports:
            if imp not in self.allowed_modules:
                return False, f"Import of '{imp}' is not allowed"

        return True, "Code is safe"

    def get_effective_code(self, code: str) -> str:
        """Get the code that will actually be executed in the sandbox."""
        return code if not self._successful_code else f"{self._successful_code}\n\n{code}"

    async def close(self):
        """Close sandbox resources."""
        return None

    @contextmanager
    def _create_safe_environment(self):
        """Create safe execution environment with temporary directory"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")

        try:
            # Create safe Python script
            script_path = os.path.join(temp_dir, "code.py")

            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"

            yield script_path, env, temp_dir

        finally:
            # Clean up temporary directory
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    async def execute_code(self, code: str) -> str:
        """Execute Python code in sandbox with safety checks"""
        # Check memory usage before execution
        current_memory = get_memory_usage()
        if current_memory > TOOL_CONFIGS["max_memory_usage"]:
            aggressive_cleanup_memory()
            return "Memory usage too high, please try again"

        # Check code safety
        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return message

        previous_code = self._successful_code
        combined_code = self.get_effective_code(code)
        default_import_block = self._build_default_import_block()

        # Replay prior successful code silently, then run only the new code with captured output.
        indented_previous_code = "\n".join("    " + line for line in previous_code.split("\n")) if previous_code else ""
        indented_current_code = "\n".join("    " + line for line in code.split("\n"))

        wrapped_code = f"""import sys
import traceback
from io import StringIO
import resource

{default_import_block}

# Set memory limit (4GB)
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
except Exception:
    pass

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()

try:
    if {bool(previous_code)}:
        silent_stdout = StringIO()
        silent_stderr = StringIO()
        sys.stdout = silent_stdout
        sys.stderr = silent_stderr
{indented_previous_code if previous_code else '        pass'}

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    # Current user code
{indented_current_code}
    
    # Get output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return result
    result = ""
    if stdout_output:
          result += stdout_output
    if stderr_output:
          if result and not result.endswith("\\n"):
                result += "\\n"
          result += stderr_output
    
    print(result)
    
except Exception as e:
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return error information
    error_msg = f"{{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
    print(error_msg)
    sys.exit(1)"""

        with self._create_safe_environment() as (script_path, env, temp_dir):
            # Write code to file
            with open(script_path, "w") as f:
                f.write(wrapped_code)

            def _run_subprocess() -> str:
                """Synchronous subprocess runner — must NOT touch the asyncio
                loop. Called via asyncio.to_thread so the blocking
                Popen/communicate/kill sequence runs on a worker thread and
                doesn't stall (or, with uvloop, abort) the event-loop thread.
                """
                try:
                    process = subprocess.Popen(
                        ["python3", script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                        cwd=temp_dir,
                        text=True,
                    )
                    try:
                        stdout, stderr = process.communicate(timeout=self.timeout)
                        if process.returncode == 0:
                            return ("ok", stdout.strip())
                        else:
                            err = f"Process exited with code {process.returncode}\n{stderr}"
                            if stdout.strip():
                                err = stdout.strip()
                            return ("err", err)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        try:
                            process.communicate(timeout=5)
                        except Exception:
                            pass
                        return ("err", f"Code execution timed out after {self.timeout} seconds")
                except Exception as e:
                    return ("err", f"Failed to execute code: {str(e)}")

            try:
                kind, payload = await asyncio.to_thread(_run_subprocess)
                if kind == "ok":
                    self._successful_code = combined_code
                result = payload
            except Exception as e:
                result = f"Failed to execute code: {str(e)}"

            # Check memory usage after execution and cleanup if needed
            cleanup_message = check_and_cleanup_memory()
            if cleanup_message:
                print(f"Memory cleanup: {cleanup_message}")

            return result


class JupyterPythonSandbox(PythonSandbox):
    """Stateful Python sandbox backed by a dedicated Jupyter kernel."""

    def __init__(self, timeout: int = 300, memory_limit: str = "100MB"):
        super().__init__(timeout=timeout, memory_limit=memory_limit)
        self._kernel_manager = None
        self._kernel_client = None
        self._kernel_lock = asyncio.Lock()
        self._temp_dir = None

    def get_effective_code(self, code: str) -> str:
        """Jupyter executes only the current cell against persistent kernel state."""
        return code

    async def _ensure_kernel(self):
        if self._kernel_client is not None:
            return
        if AsyncKernelManager is None:
            raise RuntimeError("jupyter_client is required for TOOL_SANDBOX_BACKEND=jupyter")

        self._temp_dir = tempfile.mkdtemp(prefix="python_jupyter_sandbox_")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._kernel_manager = AsyncKernelManager(kernel_name="python3")
        await self._kernel_manager.start_kernel(cwd=self._temp_dir, env=env)
        self._kernel_client = self._kernel_manager.client()
        self._kernel_client.start_channels()
        await self._kernel_client.wait_for_ready(timeout=self.timeout)
        bootstrap_code = self._build_default_import_block()
        msg_id = self._kernel_client.execute(bootstrap_code, stop_on_error=True)
        while True:
            msg = await asyncio.wait_for(self._kernel_client.get_iopub_msg(), timeout=self.timeout)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})
            if msg_type == "error":
                traceback_text = "\n".join(content.get("traceback", []))
                raise RuntimeError(
                    f"Failed to initialize default sandbox imports: {content.get('evalue', '')}\n{traceback_text}"
                )
            if msg_type == "status" and content.get("execution_state") == "idle":
                break

    @staticmethod
    def _format_rich_output(content: dict[str, Any]) -> str:
        data = content.get("data", {})
        if "text/plain" in data:
            text = data["text/plain"]
            return text if isinstance(text, str) else "".join(text)
        return ""

    async def execute_code(self, code: str) -> str:
        current_memory = get_memory_usage()
        if current_memory > TOOL_CONFIGS["max_memory_usage"]:
            aggressive_cleanup_memory()
            return "Memory usage too high, please try again"

        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return message

        async with self._kernel_lock:
            try:
                await self._ensure_kernel()
            except Exception as exc:
                return f"Failed to start Jupyter kernel: {exc}"

            stdout_parts = []
            stderr_parts = []
            rich_output_parts = []
            error_output = None

            msg_id = self._kernel_client.execute(code, stop_on_error=True)

            try:
                while True:
                    msg = await asyncio.wait_for(self._kernel_client.get_iopub_msg(), timeout=self.timeout)
                    if msg.get("parent_header", {}).get("msg_id") != msg_id:
                        continue

                    msg_type = msg.get("msg_type")
                    content = msg.get("content", {})
                    if msg_type == "stream":
                        if content.get("name") == "stderr":
                            stderr_parts.append(content.get("text", ""))
                        else:
                            stdout_parts.append(content.get("text", ""))
                    elif msg_type in {"display_data", "execute_result"}:
                        formatted = self._format_rich_output(content)
                        if formatted:
                            rich_output_parts.append(formatted)
                    elif msg_type == "error":
                        traceback_text = "\n".join(content.get("traceback", []))
                        error_output = f"{content.get('evalue', '')}\nTraceback:\n{traceback_text}"
                    elif msg_type == "status" and content.get("execution_state") == "idle":
                        break
            except asyncio.TimeoutError:
                await self._kernel_manager.interrupt_kernel()
                return f"Code execution timed out after {self.timeout} seconds"

            cleanup_message = check_and_cleanup_memory()
            if cleanup_message:
                stderr_parts.append(f"Memory cleanup: {cleanup_message}\n")

            if error_output is not None:
                return error_output

            result = ""
            stdout_output = "".join(stdout_parts)
            stderr_output = "".join(stderr_parts)
            rich_output = "\n".join(part for part in rich_output_parts if part)

            if stdout_output:
                result += stdout_output
            if rich_output:
                if result:
                    result += "\n"
                result += rich_output
            if stderr_output:
                if result:
                    result += "\n"
                result += stderr_output

            return result.strip()

    async def close(self):
        async with self._kernel_lock:
            if self._kernel_client is not None:
                self._kernel_client.stop_channels()
                self._kernel_client = None
            if self._kernel_manager is not None:
                await self._kernel_manager.shutdown_kernel(now=True)
                self._kernel_manager = None
            if self._temp_dir is not None:
                try:
                    import shutil

                    shutil.rmtree(self._temp_dir)
                except Exception:
                    pass
                self._temp_dir = None


class ToolRegistry:
    """Tool registry, manages available tools and their execution"""

    def __init__(self, backend: str | None = None):
        self.tools = {}
        self.session_id = uuid.uuid4().hex[:8]
        self.backend = (backend or TOOL_CONFIGS["sandbox_backend"]).lower()
        if self.backend == "jupyter":
            self.python_sandbox = JupyterPythonSandbox(
                timeout=TOOL_CONFIGS["jupyter_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
            )
        elif self.backend == "subprocess":
            self.python_sandbox = PythonSandbox(
                timeout=TOOL_CONFIGS["python_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
            )
        else:
            raise ValueError(f"Unsupported sandbox backend: {self.backend}")
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools in the registry"""
        # Python code interpreter
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for executing Python code in a stateful Jupyter notebook. Use print() to see output.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute"}},
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        """Register a new tool in the registry"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get all tool specifications as a list"""
        return list(self.tools.values())

    def get_effective_code(self, code: str) -> str:
        """Get the effective Python code after attaching prior successful state."""
        return self.python_sandbox.get_effective_code(code)

    async def close(self):
        """Close sandbox resources associated with this registry."""
        await self.python_sandbox.close()

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call with the given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        async with SEMAPHORE:
            if tool_name == "code_interpreter":
                return await self._execute_python(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"

        # Execute code in sandbox
        result = await self.python_sandbox.execute_code(code)
        return result


# Global tool registry instance
tool_registry = ToolRegistry()
