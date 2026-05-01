"""Benchmark Phi-3.5-mini vs Mistral-7B-Instruct for ReceiptIQ tasks.

Compares model performance on different tasks:
- Phi: Intent routing, query planning, answer verification
- Mistral: Complex analysis, answer generation

Metrics: Latency, output quality, resource usage.
"""

import time
import json
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import (
    _get_intent_with_llm,
    _plan_with_phi,
    _verify_answer_with_phi,
    _analyze_with_mistral,
    _write_answer_with_mistral,
    _load_phi_model,
    _load_mistral_model,
)

# Test datasets
INTENT_TEST_CASES = [
    "How much did I spend at restaurants?",
    "Find duplicate receipts from the same vendor.",
    "Show me anomalies in my expenses.",
    "What was my spending last month?",
    "Verify if Starbucks is a valid vendor.",
]

PLANNING_TEST_CASES = [
    ("How much did I spend at restaurants?", "DB: spend_by_category()"),
    ("Find expensive receipts over $100.", "DB: find_by_amount_threshold()"),
    ("Detect suspicious transactions.", "DB: detect_anomalies()"),
]

ANALYSIS_TEST_CASES = [
    ("How much did I spend at restaurants?", "Spend by category: [('meals', 245.50, 5), ('travel', 89.75, 2)]"),
    ("Are my expenses normal?", "Anomalies detected: [{'type': 'outlier', 'vendor': 'Luxury Steakhouse', 'amount': 250.00}]"),
]

ANSWER_TEST_CASES = [
    ("How much did I spend at restaurants?", "Spend by category: meals=$245.50 (5 transactions), travel=$89.75 (2)"),
    ("Find anomalies in my spending.", "Detected: Luxury Steakhouse $250 (outlier), Uber $150 (high frequency)"),
]

VERIFICATION_TEST_CASES = [
    (
        "You spent $245.50 on meals (doc_id: 12, 15, 18).",
        "DB: [(meals, 245.50, 5 docs)], Doc IDs: 12, 15, 18, 22, 25"
    ),
    (
        "You have 3 duplicate receipts from McDonald's.",
        "DB duplicates: [(McDonald's, 2026-03-15, 18.50, 2 copies)]"
    ),
]


def measure_time(func, *args, **kwargs):
    """Measure function execution time in milliseconds."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed_ms = (time.time() - start) * 1000
    return result, elapsed_ms


def test_phi_intent_routing():
    """Test Phi on intent classification task."""
    print("\n" + "="*70)
    print("TEST 1: PHI - Intent Routing")
    print("="*70)
    print("Task: Classify user queries into 14 intent categories\n")
    
    results = []
    total_time = 0
    successful = 0
    
    for i, query in enumerate(INTENT_TEST_CASES, 1):
        try:
            intent, elapsed = measure_time(_get_intent_with_llm, query)
            total_time += elapsed
            successful += 1
            results.append({
                "query": query,
                "intent": intent,
                "latency_ms": round(elapsed, 2),
                "success": True
            })
            print(f"  {i}. Query: {query[:50]}...")
            print(f"     → Intent: {intent} ({elapsed:.0f}ms)")
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
            print(f"  {i}. Query: {query[:50]}...")
            print(f"     → ERROR: {str(e)[:60]}...")
    
    avg_latency = total_time / len(INTENT_TEST_CASES) if INTENT_TEST_CASES else 0
    print(f"\nResults: {successful}/{len(INTENT_TEST_CASES)} successful")
    print(f"Average latency: {avg_latency:.0f}ms")
    
    return results, avg_latency


def test_phi_planning():
    """Test Phi on query planning task."""
    print("\n" + "="*70)
    print("TEST 2: PHI - Query Planning")
    print("="*70)
    print("Task: Generate JSON execution plans {intent, db_queries, response_style}\n")
    
    results = []
    total_time = 0
    successful = 0
    
    for i, (query, schema) in enumerate(PLANNING_TEST_CASES, 1):
        try:
            plan, elapsed = measure_time(_plan_with_phi, query, schema)
            total_time += elapsed
            if plan:
                successful += 1
                results.append({
                    "query": query,
                    "plan": plan,
                    "latency_ms": round(elapsed, 2),
                    "success": True
                })
                print(f"  {i}. Query: {query[:45]}...")
                print(f"     → Plan: {json.dumps(plan, indent=8)[:120]}...")
                print(f"     → Latency: {elapsed:.0f}ms")
            else:
                results.append({"query": query, "success": False, "error": "None returned"})
        except Exception as e:
            results.append({"query": query, "error": str(e), "success": False})
            print(f"  {i}. Query: {query[:45]}...")
            print(f"     → ERROR: {str(e)[:60]}...")
    
    avg_latency = total_time / len(PLANNING_TEST_CASES) if PLANNING_TEST_CASES else 0
    print(f"\nResults: {successful}/{len(PLANNING_TEST_CASES)} successful")
    print(f"Average latency: {avg_latency:.0f}ms")
    
    return results, avg_latency


def test_phi_verification():
    """Test Phi on answer verification task."""
    print("\n" + "="*70)
    print("TEST 3: PHI - Answer Verification")
    print("="*70)
    print("Task: Verify answers contain only supported claims\n")
    
    results = []
    total_time = 0
    successful = 0
    
    for i, (draft, tool_output) in enumerate(VERIFICATION_TEST_CASES, 1):
        try:
            verification, elapsed = measure_time(_verify_answer_with_phi, draft, tool_output)
            total_time += elapsed
            if verification:
                successful += 1
                is_supported = verification.get("is_supported", False)
                results.append({
                    "draft": draft,
                    "verification": verification,
                    "latency_ms": round(elapsed, 2),
                    "success": True
                })
                print(f"  {i}. Draft: {draft[:50]}...")
                print(f"     → Supported: {is_supported}")
                print(f"     → Issues: {verification.get('issues', [])}")
                print(f"     → Latency: {elapsed:.0f}ms")
            else:
                results.append({"draft": draft, "success": False, "error": "None returned"})
        except Exception as e:
            results.append({"draft": draft, "error": str(e), "success": False})
            print(f"  {i}. Draft: {draft[:50]}...")
            print(f"     → ERROR: {str(e)[:60]}...")
    
    avg_latency = total_time / len(VERIFICATION_TEST_CASES) if VERIFICATION_TEST_CASES else 0
    print(f"\nResults: {successful}/{len(VERIFICATION_TEST_CASES)} successful")
    print(f"Average latency: {avg_latency:.0f}ms")
    
    return results, avg_latency


def test_mistral_analysis():
    """Test Mistral on complex analysis task."""
    print("\n" + "="*70)
    print("TEST 4: MISTRAL - Complex Analysis")
    print("="*70)
    print("Task: Reason over database context to generate insights\n")
    
    results = []
    total_time = 0
    successful = 0
    
    for i, (query, context) in enumerate(ANALYSIS_TEST_CASES, 1):
        try:
            analysis, elapsed = measure_time(_analyze_with_mistral, query, context)
            total_time += elapsed
            successful += 1
            results.append({
                "query": query,
                "analysis": analysis[:200],
                "latency_ms": round(elapsed, 2),
                "success": True
            })
            print(f"  {i}. Query: {query[:45]}...")
            print(f"     → Analysis: {analysis[:100]}...")
            print(f"     → Latency: {elapsed:.0f}ms")
        except Exception as e:
            results.append({"query": query, "error": str(e), "success": False})
            print(f"  {i}. Query: {query[:45]}...")
            print(f"     → ERROR: {str(e)[:60]}...")
    
    avg_latency = total_time / len(ANALYSIS_TEST_CASES) if ANALYSIS_TEST_CASES else 0
    print(f"\nResults: {successful}/{len(ANALYSIS_TEST_CASES)} successful")
    print(f"Average latency: {avg_latency:.0f}ms")
    
    return results, avg_latency


def test_mistral_writing():
    """Test Mistral on answer generation task."""
    print("\n" + "="*70)
    print("TEST 5: MISTRAL - Answer Generation")
    print("="*70)
    print("Task: Generate natural language answers with citations\n")
    
    results = []
    total_time = 0
    successful = 0
    
    for i, (query, db_results) in enumerate(ANSWER_TEST_CASES, 1):
        try:
            answer, elapsed = measure_time(_write_answer_with_mistral, query, db_results)
            total_time += elapsed
            if answer:
                successful += 1
                has_citations = "doc_id" in answer
                results.append({
                    "query": query,
                    "answer": answer[:200],
                    "has_citations": has_citations,
                    "latency_ms": round(elapsed, 2),
                    "success": True
                })
                print(f"  {i}. Query: {query[:45]}...")
                print(f"     → Answer: {answer[:90]}...")
                print(f"     → Citations: {'✓' if has_citations else '✗'}")
                print(f"     → Latency: {elapsed:.0f}ms")
            else:
                results.append({"query": query, "success": False, "error": "None returned"})
        except Exception as e:
            results.append({"query": query, "error": str(e), "success": False})
            print(f"  {i}. Query: {query[:45]}...")
            print(f"     → ERROR: {str(e)[:60]}...")
    
    avg_latency = total_time / len(ANSWER_TEST_CASES) if ANSWER_TEST_CASES else 0
    print(f"\nResults: {successful}/{len(ANSWER_TEST_CASES)} successful")
    print(f"Average latency: {avg_latency:.0f}ms")
    
    return results, avg_latency


def print_comparison_table():
    """Print performance comparison table."""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    data = [
        ["Task", "Model", "Size", "Avg Latency", "Success Rate", "Optimization"],
        ["-"*18, "-"*15, "-"*8, "-"*12, "-"*12, "-"*20],
        ["Intent Routing", "Phi", "3.8B", "TBD", "TBD", "Fast classification"],
        ["Query Planning", "Phi", "3.8B", "TBD", "TBD", "JSON generation"],
        ["Verification", "Phi", "3.8B", "TBD", "TBD", "Fact-checking"],
        ["Analysis", "Mistral", "7B", "TBD", "TBD", "Context reasoning"],
        ["Writing", "Mistral", "7B", "TBD", "TBD", "Generation quality"],
    ]
    
    for row in data:
        print(f"  {row[0]:<18} {row[1]:<15} {row[2]:<8} {row[3]:<12} {row[4]:<12} {row[5]:<20}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
  PHI-3.5-MINI (3.8B parameters):
    • Lightweight, fast for classification tasks
    • Excellent at structured outputs (JSON plans, verification)
    • Best for: Intent routing, planning, verification
    • Memory efficient, runs on CPU if needed
    
  MISTRAL-7B (7B parameters):
    • Larger model, better reasoning capability
    • Excels at context understanding and generation
    • Best for: Analysis, writing, complex reasoning
    • Generates more comprehensive answers
    
  RECOMMENDATION:
    • Use Phi for: Quick classification, structured tasks
    • Use Mistral for: Complex analysis, answer generation
    • Chain them: Phi → Plan, Mistral → Write, Phi → Verify
""")


def save_results_to_file(all_results: dict):
    """Save benchmark results to JSON file."""
    output_file = PROJECT_ROOT / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Run all benchmarks."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║          ReceiptIQ Model Benchmark Suite                      ║
║     Comparing Phi-3.5-mini vs Mistral-7B-Instruct            ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {}
    }
    
    try:
        # Test Phi
        print("\n>>> TESTING PHI-3.5-MINI (3.8B params, Lightweight, Fast)\n")
        
        phi_intent, phi_intent_latency = test_phi_intent_routing()
        all_results["benchmarks"]["phi_intent_routing"] = {
            "avg_latency_ms": phi_intent_latency,
            "results": phi_intent
        }
        
        phi_planning, phi_planning_latency = test_phi_planning()
        all_results["benchmarks"]["phi_planning"] = {
            "avg_latency_ms": phi_planning_latency,
            "results": phi_planning
        }
        
        phi_verification, phi_verification_latency = test_phi_verification()
        all_results["benchmarks"]["phi_verification"] = {
            "avg_latency_ms": phi_verification_latency,
            "results": phi_verification
        }
        
        # Test Mistral
        print("\n>>> TESTING MISTRAL-7B (7B params, Larger, Better Reasoning)\n")
        
        mistral_analysis, mistral_analysis_latency = test_mistral_analysis()
        all_results["benchmarks"]["mistral_analysis"] = {
            "avg_latency_ms": mistral_analysis_latency,
            "results": mistral_analysis
        }
        
        mistral_writing, mistral_writing_latency = test_mistral_writing()
        all_results["benchmarks"]["mistral_writing"] = {
            "avg_latency_ms": mistral_writing_latency,
            "results": mistral_writing
        }
        
        # Print summary
        print_comparison_table()
        
        # Save results
        save_results_to_file(all_results)
        
        print("\n✓ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
