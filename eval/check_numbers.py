"""Every number in the report PDF (before References) must occur in a results/*.txt log after
normalising formatting. Prints the numbers that do not, for manual review.
Run: python3 -I eval/check_numbers.py"""
import re, subprocess, pathlib
D = pathlib.Path(__file__).resolve().parent.parent
text = subprocess.run(["pdftotext", str(D / "report/report.pdf"), "-"], capture_output=True, text=True).stdout
text = text.split("\nReferences\n")[0]
# Exclude this script's own output: it lists the unmatched numbers, so reading it back would
# make every previously unmatched number "match" on the next run (bug caught in the session).
logs = " ".join(p.read_text(errors="ignore") for p in list((D / "results").glob("*.txt")) + list((D / "results").glob("*.json"))
                if p.name != "check_numbers_output.txt")
norm = lambda x: x.replace(",", "")
lognums = set()
for m in re.findall(r"\d[\d,]*\.?\d*", logs):
    x = norm(m).rstrip(".")
    lognums.add(x)
    if "." in x:  # also accept the same value rounded to fewer decimals, and as a percentage
        v = float(x)
        for d in range(0, 5):
            lognums.add(f"{v:.{d}f}"); lognums.add(f"{100 * v:.{d}f}")
nums = [norm(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
missing = sorted({n for n in nums if n not in lognums and not (n.isdigit() and int(n) <= 20)}, key=lambda x: float(x))
print(f"numbers in report body: {len(nums)}, distinct: {len(set(nums))}; not found in results logs ({len(missing)}): {missing}")
