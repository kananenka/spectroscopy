import sys
sys.path.insert(0, "/work/akanane/sw/spectroscopy/")

import spec

input_file = "input_stretch_unc.json"
spec.run(input_file)
