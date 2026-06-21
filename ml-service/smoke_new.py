import urllib.request, json

def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return f"ERROR: {e}"

print("=== /api/violations/meta ===")
d = get("http://localhost:8000/api/violations/meta")
if isinstance(d, dict):
    print("  totalCases:", d["totalCases"])
    print("  topStations count:", len(d["topStations"]))
    print("  topViolationTypes count:", len(d["topViolationTypes"]))
    print("  statusCounts:", d["statusCounts"])
else: print(d)

print("=== /api/violations/stations ===")
d = get("http://localhost:8000/api/violations/stations")
if isinstance(d, list): print(f"  {len(d)} stations. First:", d[0])
else: print(d)

print("=== /api/violations/violation-types ===")
d = get("http://localhost:8000/api/violations/violation-types")
if isinstance(d, list): print(f"  {len(d)} types:", d[:4])
else: print(d)

print("=== /api/violations/type-distribution ===")
d = get("http://localhost:8000/api/violations/type-distribution")
if isinstance(d, dict): print(f"  {len(d)} types. Top3:", list(d.items())[:3])
else: print(d)

print("=== /api/violations/station-workload ===")
d = get("http://localhost:8000/api/violations/station-workload?limit=3")
if isinstance(d, list): print(f"  {len(d)} stations. First:", d[0])
else: print(d)

print("ALL NEW ENDPOINTS OK")
