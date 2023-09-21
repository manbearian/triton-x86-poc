# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner

# name: The name of this test suite
config.name = 'TRITON-X86-POC'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.triton_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# Searches for a runtime library with the given name and returns a tool
# substitution of the same name and the found path.
# Correctly handles the platforms shared library directory and naming conventions.
def add_runtime(name):
    path = ""
    for prefix in ["", "lib"]:
        path = os.path.join(
            # FIXME: hardcode shared library extension to ".so" as work around
            config.llvm_lib_dir, f"{prefix}{name}.so"
        )
        if os.path.isfile(path):
            break
    return ToolSubst(f"%{name}", path)

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs',
    'Examples',
    'CMakeLists.txt',
    'README.txt',
    'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.triton_obj_root, 'test')
config.triton_tools_dir = os.path.join(config.triton_x86_poc_obj_root, 'tools/triton-x86-poc-opt')
config.filecheck_dir = os.path.join(config.triton_obj_root, 'bin', 'FileCheck')
tool_dirs = [
    config.triton_tools_dir,
    config.llvm_tools_dir,
    config.filecheck_dir]

# Tweak the PATH to include the tools dir.
for d in tool_dirs:
    llvm_config.with_environment('PATH', d, append_path=True)
tools = [
    'triton-x86-poc-opt',
    'mlir-cpu-runner',
    add_runtime('mlir_runner_utils'),
    add_runtime('mlir_c_runner_utils'),
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# TODO: what's this?
llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_binary_dir, 'python_packages', 'triton'),
], append_path=True)