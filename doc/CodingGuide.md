# Coding and Workflow Guidelines

## Editing C files

When editing C files in Visual Studio Code, we follow the [Linux Kernel Coding Style](https://www.kernel.org/doc/html/v4.10/process/coding-style.html).
For consistency and readability, please use settings for C defined in `.vscode/settings.json`. To use these settings in VS Code, simply open the directory by `code tinyRuntime` to automatically apply these settings or copy content to your own `.vscode` folder.

## Editing Python files

We don't have a strict coding style in Python, but please ensure your code is clean and readable. In this repository, we primarily work with Python using Jupyter notebooks due to their convenience for development. However, to make the implemented functions reusable across other notebooks and to easily track changes in commits, please convert the notebook to a Python file using `./nbconvert example.ipynb`. This will generate `example.py`. Commit both the notebook and the Python file to the repository.

