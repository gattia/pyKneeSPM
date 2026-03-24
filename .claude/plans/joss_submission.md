# Plan: JOSS Paper Submission

## Overview

Publish pyKneeSPM (to be renamed pyMeshSPM) as a short paper in the Journal of Open Source Software. The library is generic — it works on any triangulated surface mesh with scalar data — so the paper should frame it broadly (musculoskeletal, dental, cranial, etc.) with knee cartilage thinning from OAI as the motivating use case and validation example.

## TODO Before Submission

### 1. Rename to pyMeshSPM
- Rename package directory `pyKneeSPM/` → `pyMeshSPM/`
- Update `pyproject.toml` (project name, package find)
- Update all internal imports
- Update `__init__.py`, `CLAUDE.md`, `README.MD`
- Rename GitHub repo `pyKneeSPM` → `pyMeshSPM`
- Update any references to "knee" in docstrings/comments that imply the tool is knee-specific (keep knee references in examples where appropriate)

### 2. Example data from OAI
- Prepare a downloadable example dataset from the OAI cartilage thinning rate mapping work currently in progress
- Host on Zenodo, figshare, or similar (JOSS expects archived data with a DOI)
- Include: pre-aligned meshes with cartilage thickness scalars for a subset of participants, group labels, reference mesh
- Keep the download size reasonable (subset of participants, single bone surface)
- Add a download script or function (e.g., `pyMeshSPM.fetch_example_data()`) that pulls the dataset on first use

### 3. PyPI publishing
- Set up PyPI account and package
- Configure `pyproject.toml` for PyPI (version, URLs, classifiers)
- Add GitHub Actions workflow for automated publishing on release tags
- Test with `pip install pymeshspm` from a clean environment
- Add badge to README

### 4. OAI examples
- Add example scripts/notebooks that:
  - Load the OAI example data
  - Run a two-sample comparison of cartilage thinning rates between groups (e.g., OA progressors vs non-progressors)
  - Visualize the results (t-statistic map, significant clusters on the mesh)
  - Run the one-sample z-test on change maps
- Examples should be runnable end-to-end with `pip install pymeshspm` + the example data download
- Put in an `examples/` directory

### 5. JOSS paper
- Write `paper.md` following JOSS format (title, authors, summary, statement of need, references)
- Key points to cover:
  - Gap: no existing open-source tool for surface-mesh SPM in musculoskeletal imaging
  - Approach: permutation-based cluster-extent inference on registered bone surfaces
  - Generality: works on any triangulated mesh, not knee-specific
  - Validation: OAI cartilage thinning example reproducing expected group differences
- Include a figure showing the pipeline (mesh → statistic map → clusters → significance)
- Add `paper.bib` with references (neuroimaging SPM origins, OAI dataset, pyfocusr, relevant MSK imaging papers)
- Ensure JOSS checklist items are met: documentation, tests, installation instructions, example, contribution guidelines, license

### 6. Pre-submission cleanup
- Add `CONTRIBUTING.md`
- Add `CODE_OF_CONDUCT.md`
- Review and resolve any remaining known bugs from CLAUDE.md
- Ensure all tests pass in a clean environment
- Add CI (GitHub Actions) for running tests on push
