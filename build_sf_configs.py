#!/usr/bin/env python3
"""Build the ActivitySim configuration with the weight columns for the San Francisco Bay Area example."""
import os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "sf")
MEL = os.path.join(HERE, "melb", "configs")
SRC, DST = os.path.join(FULL, "configs"), os.path.join(FULL, "configs_bpir")

assert os.path.isdir(SRC), f"stock configs not found: {SRC} — download the example first"
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
for f in ["annotate_households.csv", "annotate_landuse.csv",
          "cdap_indiv_and_hhsize1.csv", "cdap_coefficients.csv", "logging.yaml"]:
    shutil.copyfile(os.path.join(MEL, f), os.path.join(DST, f))
    print("hooked:", f)

p = os.path.join(DST, "settings.yaml")
s = open(p).read()
s = s.replace("      - num_workers\n", "      - num_workers\n      - w_income\n      - w_vot\n", 1)
s = s.replace("      - ptype\n", "      - ptype\n      - cdap_work_w\n      - cdap_student_w\n      - cdap_nonmand_w\n", 1)
s = s.replace("      - TERMINAL\n", "      - TERMINAL\n      - w_parking\n      - w_terminal\n      - w_employment\n", 1)
s = re.sub(r"households_sample_size:\s*\d+", "households_sample_size: 0", s, count=1)
s += "\n# BPIR full-scale demo\ntrace_hh_id:\ntrace_od:\n"
open(p, "w").write(s)
for probe in ["w_vot", "cdap_nonmand_w", "w_employment", "households_sample_size: 0", "trace_hh_id:"]:
    assert probe in open(p).read(), probe
print("settings.yaml patched")

p = os.path.join(DST, "network_los.yaml")
s = open(p).read()
s = s.replace("taz_skims: skims.omx", "taz_skims: omx/skims.omx")
s = s.replace("write_skim_cache: True", "write_skim_cache: False")
open(p, "w").write(s)
assert "taz_skims: omx/skims.omx" in open(p).read()
print("network_los.yaml patched — configs_bpir ready")
