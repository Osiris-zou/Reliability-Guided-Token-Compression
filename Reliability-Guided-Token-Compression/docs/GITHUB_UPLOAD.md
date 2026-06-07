# Uploading this repository to GitHub

Recommended repository name:

```text
Reliability-Guided-Token-Compression
```

From the parent directory of this folder, run:

```bash
cd Reliability-Guided-Token-Compression
git init
git add .
git commit -m "Reproducibility code for reliability-guided token compression"
git branch -M main
git remote add origin https://github.com/Osiris-zou/Reliability-Guided-Token-Compression.git
git push -u origin main
```

If you prefer to reuse the old `ACEC_main` repository, update all URLs in `README.md`, `CITATION.cff`, and `docs/CODE_AVAILABILITY_TEXT.md` before pushing. However, a new repository name is recommended because the manuscript title has changed.

Do not upload datasets, checkpoints, generated images, result logs, or local output files. The `.gitignore` file excludes `outputs/`, `results/`, checkpoint formats, and common log/artifact files.
