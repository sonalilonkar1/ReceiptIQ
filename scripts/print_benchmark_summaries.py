import glob, json

files = sorted(glob.glob("outputs/benchmark_summary_*.json"))[-4:]
print("file,overall_success,avg_latency_ms,citation_rate,tool_usage_rate,injection_refusal")

for f in files:
    d = json.load(open(f))
    s = d.get("statistics", d)
    print(
        f.split("/")[-1],
        s.get("overall_success_rate"),
        s.get("avg_latency_ms"),
        s.get("citation_rate"),
        s.get("tool_usage_rate"),
        s.get("injection_refusal_rate"),
    )