import nbformat

notebook_path = "SEOHackathonChallenge.ipynb"

nb = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)

# Fix or remove metadata.widgets
if "widgets" in nb.metadata:
    if "state" not in nb.metadata["widgets"]:
        nb.metadata["widgets"]["state"] = {}  # Or: del nb.metadata["widgets"]

nbformat.write(nb, notebook_path)
