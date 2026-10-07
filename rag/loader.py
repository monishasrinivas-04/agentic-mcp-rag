from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
}


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}


def load_repository(repo_path):

    documents = []

    repo_path = Path(repo_path)

    for file_path in repo_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if any(
            directory in file_path.parts
            for directory in IGNORED_DIRECTORIES
        ):
            continue

        try:

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            if content.strip():

                documents.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "extension": file_path.suffix,
                    "content": content
                })

        except Exception as e:

            print(f"Could not read {file_path}: {e}")

    return documents