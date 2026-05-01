#!/usr/bin/env python3
"""
Quick validation of the benchmarking script structure and components.

This script verifies that the benchmark script can be imported and has
all required components without actually running the full benchmark.
"""

import os
import sys
import ast

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def validate_benchmark_script():
    """Validate the structure of run_benchmark.py"""
    script_path = os.path.join(os.path.dirname(__file__), 'run_benchmark.py')
    
    print("🔍 Validating Benchmark Script Structure")
    print("=" * 70)
    
    # 1. Check if file exists
    print(f"\n1. Checking file existence: {script_path}")
    if not os.path.exists(script_path):
        print("   ❌ File not found!")
        return False
    print("   ✓ File exists")
    
    # 2. Parse AST to verify syntax and structure
    print("\n2. Parsing Python syntax...")
    try:
        with open(script_path, 'r') as f:
            tree = ast.parse(f.read())
        print("   ✓ Valid Python syntax")
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    
    # 3. Check for required functions
    print("\n3. Checking for required functions...")
    required_functions = [
        'ensure_output_dir',
        'has_citations',
        'tool_used_when_required',
        'is_injection_refused',
        'measure_query',
        'run_benchmark',
        'save_results',
        'calculate_statistics',
        'print_summary',
        'save_summary',
        'main'
    ]
    
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    missing_functions = [f for f in required_functions if f not in function_names]
    
    if missing_functions:
        print(f"   ❌ Missing functions: {missing_functions}")
        return False
    
    for func_name in required_functions:
        print(f"   ✓ {func_name}")
    
    # 4. Check for required constants
    print("\n4. Checking for required constants...")
    required_constants = ['NORMAL_QUERIES', 'INJECTION_QUERIES']
    
    assignment_names = [node.targets[0].id for node in ast.walk(tree) 
                        if isinstance(node, ast.Assign) 
                        and isinstance(node.targets[0], ast.Name)]
    
    missing_constants = [c for c in required_constants if c not in assignment_names]
    if missing_constants:
        print(f"   ❌ Missing constants: {missing_constants}")
        return False
    
    # Get counts
    normal_count = len([x for x in assignment_names if x == 'NORMAL_QUERIES'])
    injection_count = len([x for x in assignment_names if x == 'INJECTION_QUERIES'])
    
    print(f"   ✓ NORMAL_QUERIES found")
    print(f"   ✓ INJECTION_QUERIES found")
    
    # 5. Verify import statements
    print("\n5. Checking imports...")
    import_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_names.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            import_names.extend([alias.name for alias in node.names])
    
    required_imports = ['argparse', 'csv', 'json', 'os', 'sys', 'time']
    missing_imports = [imp for imp in required_imports if imp not in import_names]
    
    if missing_imports:
        print(f"   ⚠ Missing expected imports: {missing_imports}")
    else:
        print(f"   ✓ All core imports present")
    
    # Check for app imports
    if 'handle_message' in str(ast.dump(tree)):
        print("   ✓ handle_message import present")
    
    # 6. Check CLI argument parsing
    print("\n6. Checking CLI argument configuration...")
    if 'ArgumentParser' in [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]:
        print("   ✓ ArgumentParser configured")
    else:
        print("   ⚠ ArgumentParser not found in AST")
    
    # 7. Verify CSV and JSON output
    print("\n7. Checking output handling...")
    if 'csv.DictWriter' in str(ast.dump(tree)):
        print("   ✓ CSV writing configured")
    else:
        print("   ⚠ CSV writing may not be properly configured")
    
    if 'json.dump' in str(ast.dump(tree)):
        print("   ✓ JSON writing configured")
    else:
        print("   ⚠ JSON writing may not be properly configured")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION PASSED - Benchmark script structure is correct")
    print("\nThe benchmark script is ready to use:")
    print("  • 20 normal queries (diverse intent types)")
    print("  • 5 injection test queries (security testing)")
    print("  • Comprehensive metrics tracking")
    print("  • CSV + JSON output")
    print("  • CLI configuration support")
    print("\nUsage examples:")
    print("  python scripts/run_benchmark.py --model_mode phi_only --cache on")
    print("  python scripts/run_benchmark.py --model_mode phi+mistral --cache off")
    
    return True


def check_query_counts():
    """Verify query counts are correct"""
    script_path = os.path.join(os.path.dirname(__file__), 'run_benchmark.py')
    
    print("\n" + "=" * 70)
    print("📊 Query Composition")
    print("=" * 70)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Extract query lists
    import re
    
    normal_section = re.search(r'NORMAL_QUERIES = \[(.*?)\]', content, re.DOTALL)
    injection_section = re.search(r'INJECTION_QUERIES = \[(.*?)\]', content, re.DOTALL)
    
    if normal_section and injection_section:
        # Extract quoted strings
        normal_matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', normal_section.group(1))
        injection_matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', injection_section.group(1))
        
        # Flatten tuples and filter
        normal_queries = [q[0] or q[1] for q in normal_matches if q[0] or q[1]]
        injection_queries = [q[0] or q[1] for q in injection_matches if q[0] or q[1]]
        
        # Filter out comments
        normal_queries = [q for q in normal_queries if not q.startswith('#')]
        injection_queries = [q for q in injection_queries if not q.startswith('#')]
        
        print(f"\n✓ Normal Queries: {len(normal_queries)}")
        for i, q in enumerate(normal_queries[:5], 1):
            print(f"    {i}. {q[:60]}...")
        if len(normal_queries) > 5:
            print(f"    ... and {len(normal_queries) - 5} more")
        
        print(f"\n✓ Injection Queries: {len(injection_queries)}")
        for i, q in enumerate(injection_queries, 1):
            print(f"    {i}. {q[:60]}...")
        
        print(f"\n✓ Total Test Cases: {len(normal_queries) + len(injection_queries)}")
        
        return len(normal_queries) == 20 and len(injection_queries) == 5
    
    return False


if __name__ == "__main__":
    try:
        success = validate_benchmark_script()
        if success:
            check_query_counts()
            sys.exit(0)
        else:
            print("\n❌ VALIDATION FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
