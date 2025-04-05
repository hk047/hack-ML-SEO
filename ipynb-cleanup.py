import nbformat

def clean_notebook(path):
    # Read notebook with safe fallback
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # Safely delete widgets metadata
    if "widgets" in nb.metadata:
        print("Removing metadata.widgets...")
        del nb.metadata["widgets"]

    # Remove widgets from *each cell* too, if any exist
    for cell in nb.cells:
        if "metadata" in cell and "widgets" in cell["metadata"]:
            print("Removing cell-level widgets...")
            del cell["metadata"]["widgets"]

    # Write back the cleaned notebook
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook cleaned: {path}")

# Usage
clean_notebook("SEOHackathonChallenge.ipynb")
