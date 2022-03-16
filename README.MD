# PyKneeSPM

Python module to perform SPM (statistical parametric mapping) style analysis on 3D surface meshes of knee bones.

# Introduction / Background
This work is inspired by the brain imaging communities use of SPM to find "activations" in the brain and the lack of 
ways to find anatomically specific information about musculoskeletal joint health. Current methods use human defined 
ROIs that may or may not include the most relevant information. This package will find clusters of significant change. 

This package works under the premise that knee information can be localized to the surface of the bone. For example, 
we can easily assign cartilage thickness or T2 values to the bone surface for statistical analysis. We then register 
the bones to a common reference frame to enable calculation of group statistics.

### Statistical Analysis
The current implementation uses a permutation approach for statistical analyses. Simple correlations & One-sample 
difference tests are available. 

# Installation

Work in progress - please file an Issue if you are here to try out the package but the installtion instructions are not updated yet. 

```bash
git clone XXXX
cd pyKneeSPM
python setup.py install
```

# Examples

Work in progress - please file an Issue if you are here to try out the package but the examples aren't posted yet. 


MIT License.
