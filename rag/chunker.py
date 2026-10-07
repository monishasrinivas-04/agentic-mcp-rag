import ast
import re


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def is_separator_line(line):
    """
    Detect lines such as:

    # -----------------------------------
    """

    stripped = line.strip()

    if not stripped.startswith("#"):
        return False

    content = stripped[1:].strip()

    return (
        len(content) >= 5
        and set(content) <= {"-"}
    )


def is_section_header(line):
    """
    Detect section headers such as:

    # PAGE CONFIG
    # LOAD YOLO MODEL
    # LOAD LST TIFF FILE
    """

    stripped = line.strip()

    if not stripped.startswith("#"):
        return False

    content = stripped[1:].strip()

    if not content:
        return False

    # Section headers in this project are written in uppercase.
    return (
        content == content.upper()
        and any(char.isalpha() for char in content)
    )


def chunk_python_by_sections(text):
    """
    Split Python files using the project's
    separator + uppercase section-header pattern.

    Example:

    # -----------------------------------
    # LOAD LST TIFF FILE
    # -----------------------------------

    becomes:

    section = "LOAD LST TIFF FILE"
    content = corresponding code
    """

    lines = text.splitlines()

    sections = []

    i = 0

    while i < len(lines) - 1:

        if (
            is_separator_line(lines[i])
            and is_section_header(lines[i + 1])
        ):

            section_name = (
                lines[i + 1]
                .strip()
                .lstrip("#")
                .strip()
            )

            start = i

            # Find the next section
            j = i + 2

            while j < len(lines) - 1:

                if (
                    is_separator_line(lines[j])
                    and is_section_header(lines[j + 1])
                ):
                    break

                j += 1

            end = j

            content = "\n".join(
                lines[start:end]
            )

            if content.strip():

                sections.append({
                    "section": section_name,
                    "content": content
                })

            i = j

        else:
            i += 1

    return sections


def chunk_python_code(text):
    """
    Python chunking strategy:

    1. Use explicit project sections if available.
    2. Otherwise use AST for functions/classes.
    3. Otherwise fall back to fixed-size chunks.
    """

    # --------------------------------------------------
    # Strategy 1: Project section headers
    # --------------------------------------------------

    sections = chunk_python_by_sections(text)

    if len(sections) >= 2:

        return sections

    # --------------------------------------------------
    # Strategy 2: AST
    # --------------------------------------------------

    try:
        tree = ast.parse(text)

    except SyntaxError:

        return [
            {
                "section": None,
                "content": chunk
            }
            for chunk in chunk_text(text)
        ]

    lines = text.splitlines()

    chunks = []

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef
            )
        ):

            start = node.lineno - 1
            end = node.end_lineno

            content = "\n".join(
                lines[start:end]
            )

            if content.strip():

                chunks.append({
                    "section": getattr(
                        node,
                        "name",
                        None
                    ),
                    "content": content
                })

    if chunks:

        return chunks

    # --------------------------------------------------
    # Strategy 3: Fixed-size fallback
    # --------------------------------------------------

    return [
        {
            "section": None,
            "content": chunk
        }
        for chunk in chunk_text(text)
    ]


def create_chunks(documents):

    chunks = []

    for document in documents:

        extension = document["extension"].lower()

        # --------------------------------------------------
        # Python files
        # --------------------------------------------------

        if extension == ".py":

            python_chunks = chunk_python_code(
                document["content"]
            )

            for chunk_id, chunk_data in enumerate(
                python_chunks
            ):

                chunks.append({

                    "file_path":
                        document["file_path"],

                    "file_name":
                        document["file_name"],

                    "extension":
                        document["extension"],

                    "chunk_id":
                        chunk_id,

                    "section":
                        chunk_data["section"],

                    "content":
                        chunk_data["content"]
                })

        # --------------------------------------------------
        # Other files
        # --------------------------------------------------

        else:

            text_chunks = chunk_text(
                document["content"]
            )

            for chunk_id, content in enumerate(
                text_chunks
            ):

                chunks.append({

                    "file_path":
                        document["file_path"],

                    "file_name":
                        document["file_name"],

                    "extension":
                        document["extension"],

                    "chunk_id":
                        chunk_id,

                    "section":
                        None,

                    "content":
                        content
                })

    return chunks