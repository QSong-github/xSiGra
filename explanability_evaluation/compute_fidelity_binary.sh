#!/bin/bash
python3 compute_fidelity_all_binary.py --benchmark gradcam
python3 compute_fidelity_mask_binary.py --benchmark gradcam
python3 compute_fidelity_all_binary.py --benchmark saliency
python3 compute_fidelity_mask_binary.py --benchmark saliency
python3 compute_fidelity_all_binary.py --benchmark deconvolution
python3 compute_fidelity_mask_binary.py --benchmark deconvolution
python3 compute_fidelity_all_binary.py --benchmark inputxgradient
python3 compute_fidelity_mask_binary.py --benchmark inputxgradient
python3 compute_fidelity_all_binary.py --benchmark guidedbackprop
python3 compute_fidelity_mask_binary.py --benchmark guidedbackprop