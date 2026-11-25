import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "dashboard"))  # `streamlit run dashboard/Overview.py` does the same
pages = [root / "dashboard" / "Overview.py", *sorted((root / "dashboard" / "pages").glob("*.py"))]
bad = 0
for p in pages:
    at = AppTest.from_file(str(p), default_timeout=60)
    at.run()
    errs = [e.value for e in at.exception] + [e.value for e in at.error]
    print(f"{p.name:<22} {'OK' if not errs else 'FAIL'}  "
          f"(markdown={len(at.markdown)}, charts={len(at.get('plotly_chart'))})")
    for e in errs:
        bad += 1
        print("   ", str(e)[:600])
sys.exit(1 if bad else 0)
