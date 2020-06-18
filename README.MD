# PyKneeSPM

Python module to perform SPM (statistical parametric mapping) style analysis on 3D surface meshes of the knee

# Introduction / Background
This work is inspired by the brain imaging communities use of SPM to find "activations" in the brain, and the lack of 
ways to find anatomically specific information about musculoskeletal joint health. Current methods use human defined 
ROIs that may or may not be key regions of interest. 

This package works under the premise that knee health information can be localized to the surface of the bone
and that by registering bones together we can complete group statistics on these metrics. Examples include
transferring cartilage thickness or T2 values to the bone surface directly.

# Installation
```
git clone XXXX
cd pyKneeSPM
python setup.py install
```

# Examples




MIT License.
