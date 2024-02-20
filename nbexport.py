# Don't edit this file! This was automatically generated from "nbexport.ipynb".

import nbformat

def nbexport(nbname):
    with open(nbname, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    cells = [c.source for c in nb.cells if c.cell_type == 'code' and '#|export' in c.source]
    cells = ["\n".join(c.split('\n')[1:]) for c in cells]
    pyname = nbname.replace('.ipynb', '.py')
    with open(pyname, 'w', encoding='utf-8') as f:
        f.write(f"# Don\'t edit this file! This was automatically generated from \"{nbname}\".\n\n")
        f.write("\n\n".join(cells))
        f.write('\n')
