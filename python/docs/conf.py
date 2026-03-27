project = "streampress"
copyright = "2024, Zachary DeBruine"
author = "Zachary DeBruine"
release = "1.0.0"

extensions = [
    "myst_nb",
]

html_theme = "furo"
html_title = "streampress"
html_theme_options = {
    "source_repository": "https://github.com/zdebruine/streampress",
    "source_branch": "main",
}

# MyST-NB: execute notebooks at build time
nb_execution_mode = "force"
nb_execution_timeout = 180
nb_kernel_rgx_aliases = {".*": "python3"}

myst_enable_extensions = ["colon_fence"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
