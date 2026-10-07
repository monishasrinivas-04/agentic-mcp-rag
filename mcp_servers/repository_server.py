from pathlib import Path

from mcp.server import MCPServer


# --------------------------------------------------
# MCP SERVER
# --------------------------------------------------

mcp = MCPServer("urban-heat-repository")


# --------------------------------------------------
# REPOSITORY LOCATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "urban-heat-repo"
)


# --------------------------------------------------
# SUPPORTED FILE TYPES
# --------------------------------------------------

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


# --------------------------------------------------
# INTERNAL FILE ITERATOR
# --------------------------------------------------

def iter_repository_files():

    for file_path in REPO_ROOT.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if any(
            directory in file_path.parts
            for directory in IGNORED_DIRECTORIES
        ):
            continue

        yield file_path


# --------------------------------------------------
# SAFE PATH VALIDATION
# --------------------------------------------------

def get_safe_path(file_path: str) -> Path:

    repository_root = REPO_ROOT.resolve()

    requested_path = (
        REPO_ROOT / file_path
    ).resolve()

    try:

        requested_path.relative_to(
            repository_root
        )

    except ValueError:

        raise ValueError(
            "Access denied: path is outside "
            "the repository."
        )

    if not requested_path.is_file():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return requested_path


# ==================================================
# MCP TOOL 1
# ==================================================

@mcp.tool()
def list_files() -> list[str]:
    """
    List readable files in the repository.
    """

    files = []

    for file_path in iter_repository_files():

        relative_path = (
            file_path.relative_to(REPO_ROOT)
        )

        files.append(
            str(relative_path).replace("\\", "/")
        )

    return sorted(files)


# ==================================================
# MCP TOOL 2
# ==================================================

@mcp.tool()
def read_file(file_path: str) -> str:
    """
    Read a specific repository file.
    """

    path = get_safe_path(file_path)

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


# ==================================================
# MCP TOOL 3
# ==================================================

@mcp.tool()
def search_repository(
    query: str,
    max_results: int = 5
) -> list[dict]:
    """
    Search repository files using keyword matching.
    """

    query = query.strip().lower()

    if not query:

        raise ValueError(
            "Search query cannot be empty."
        )

    terms = query.split()

    results = []

    for file_path in iter_repository_files():

        try:

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        except Exception:
            continue

        lower_content = content.lower()

        score = sum(
            lower_content.count(term)
            for term in terms
        )

        if score == 0:
            continue

        # Find a useful snippet
        first_term = terms[0]

        position = lower_content.find(
            first_term
        )

        if position >= 0:

            start = max(
                0,
                position - 200
            )

            end = min(
                len(content),
                position + 800
            )

            snippet = content[start:end]

        else:

            snippet = content[:1000]

        results.append(
            {
                "file_path": str(
                    file_path.relative_to(
                        REPO_ROOT
                    )
                ).replace("\\", "/"),

                "score": score,

                "snippet": snippet
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:max_results]


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    mcp.run()