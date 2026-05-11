#!/usr/bin/env python3
"""
Comprehensive guardrail verification script.
Runs all guardrail tests and prints a summary report.

Usage:
    python scripts/run_guardrail_checks.py
"""

import sys
import subprocess
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def run_command(cmd, description):
    """Run a command and return (success, output)."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{description}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        success = result.returncode == 0
        print(output)
        return success, output
    except subprocess.TimeoutExpired:
        print(f"{RED}TIMEOUT: Command took too long to execute{RESET}")
        return False, "TIMEOUT"
    except Exception as e:
        print(f"{RED}ERROR: {e}{RESET}")
        return False, str(e)


def main():
    """Run all guardrail checks."""
    project_root = Path(__file__).resolve().parents[1]
    
    print(f"\n{BOLD}{BLUE}")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "RECEIPTIQ GUARDRAIL VERIFICATION" + " "*20 + "║")
    print("║" + " "*68 + "║")
    print("║" + " "*10 + "Comprehensive test suite for edit guardrails" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    print(f"{RESET}\n")
    
    # Track results
    results = {}
    
    # Step 1: Initialize database
    print(f"{BOLD}Step 1/3: Initializing Database{RESET}")
    success, output = run_command(
        [sys.executable, str(project_root / "scripts" / "init_db.py")],
        "📦 Database Initialization"
    )
    results["init_db"] = success
    
    # Step 2: Run edit guardrails test
    print(f"\n{BOLD}Step 2/3: Running Edit Guardrails Tests{RESET}")
    success, output = run_command(
        [sys.executable, str(project_root / "scripts" / "test_edit_guardrails.py")],
        "🛡️  Edit Guardrails Validation"
    )
    results["test_edit_guardrails"] = success
    
    # Extract test count from output
    guardrails_passed = output.count("✅ PASS")
    guardrails_total = output.count("PASS") + output.count("FAIL")
    
    # Step 3: Run edit history test
    print(f"\n{BOLD}Step 3/3: Running Edit History & Error Handling Tests{RESET}")
    success, output = run_command(
        [sys.executable, str(project_root / "scripts" / "test_ui_edit_history.py")],
        "📜 Edit History UI Workflow"
    )
    results["test_ui_edit_history"] = success
    
    history_passed = output.count("✅ PASS")
    history_total = output.count("PASS") + output.count("FAIL")
    
    # Print comprehensive summary
    print(f"\n{BOLD}{BLUE}")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*23 + "FINAL VERIFICATION REPORT" + " "*19 + "║")
    print("╚" + "═"*68 + "╝")
    print(f"{RESET}\n")
    
    # Summary table
    print(f"{BOLD}Test Results:{RESET}\n")
    
    test_results = [
        ("Database Initialization", results.get("init_db", False)),
        ("Edit Guardrails Tests", results.get("test_edit_guardrails", False)),
        ("Edit History UI Tests", results.get("test_ui_edit_history", False)),
    ]
    
    all_passed = all(result for _, result in test_results)
    
    for test_name, passed in test_results:
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {status} | {test_name}")
    
    # Detailed statistics
    print(f"\n{BOLD}Detailed Statistics:{RESET}\n")
    print(f"  Edit Guardrails:    {GREEN}{guardrails_passed}/{guardrails_total} tests passed{RESET}")
    print(f"  Edit History:       {GREEN}{history_passed}/{history_total} tests passed{RESET}")
    print(f"  Total Tests:        {GREEN}{guardrails_passed + history_passed}/{guardrails_total + history_total} passed{RESET}")
    
    # Final verdict
    print(f"\n{BOLD}")
    if all_passed:
        print(f"{GREEN}{'='*70}{RESET}")
        print(f"{GREEN}🎉 ALL GUARDRAIL CHECKS PASSED!{RESET}")
        print(f"{GREEN}{'='*70}{RESET}")
        print(f"""
{GREEN}✅ Your system is verified to have:{RESET}
    ✓ Input validation guardrails (type, range, format)
    ✓ High-risk edit warnings with confirmation
    ✓ Totals mismatch detection and flagging
    ✓ Complete edit audit trails
    ✓ Pending receipt auto-clear logic
    ✓ Vendor profile learning
    ✓ Integrity checking and status reporting
    ✓ Clean error handling (no stack traces to users)
    
{GREEN}→ The system is READY FOR PRODUCTION{RESET}
        """)
        return 0
    else:
        print(f"{RED}{'='*70}{RESET}")
        print(f"{RED}❌ SOME TESTS FAILED{RESET}")
        print(f"{RED}{'='*70}{RESET}")
        print(f"""
{RED}⚠️ Please review failed tests above and fix issues{RESET}

Failed steps:
""")
        for test_name, passed in test_results:
            if not passed:
                print(f"  {RED}✗{RESET} {test_name}")
        
        print(f"\n→ Rerun this script after fixing issues:")
        print(f"  python scripts/run_guardrail_checks.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
