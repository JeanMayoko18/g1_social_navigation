"""Plot declared group totals separately from per-trial outcome annotations."""
import csv
import importlib.util
from pathlib import Path


def export_declared_counts(path, output):
    """Totals are never used to fabricate individual trial outcomes."""
    if path is None:
        return
    path=Path(path)
    with path.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    module_path=Path(__file__).resolve().parents[1]/'plot_success_counts.py'
    # The standalone plotting CLI remains independent of ROS and analysis internals.
    import subprocess,sys
    subprocess.run([sys.executable,str(module_path),'--counts',str(path),
                    '--output',str(Path(output)/'declared_success')],check=True)
    (Path(output)/'declared_success'/'success_counts_used.csv').write_bytes(path.read_bytes())
