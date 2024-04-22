#!/bin/bash
python3 compute_fidelity_all.py --benchmark gradcam
python3 compute_fidelity_mask.py --benchmark gradcam
python3 compute_fidelity_all.py --benchmark saliency
python3 compute_fidelity_mask.py --benchmark saliency
python3 compute_fidelity_all.py --benchmark deconvolution
python3 compute_fidelity_mask.py --benchmark deconvolution
python3 compute_fidelity_all.py --benchmark inputxgradient
python3 compute_fidelity_mask.py --benchmark inputxgradient
python3 compute_fidelity_all.py --benchmark guidedbackprop
python3 compute_fidelity_mask.py --benchmark guidedbackprop