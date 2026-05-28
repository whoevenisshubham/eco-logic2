from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


IGNORED_DIRS = {".git", "bin", "obj", "dist", "build", "target", "node_modules", "vendor", ".venv"}
DOTNET_SUFFIXES = {".cs", ".fs", ".vb", ".csproj", ".fsproj", ".vbproj", ".sln"}


@dataclass
class FileRecord:
    path: str
    relative_path: str
    suffix: str
    language: str
    size_bytes: int
    imports: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


def detect_language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "Python"
    if suffix in {".cpp", ".cc", ".cxx", ".hpp", ".h"}:
        return "C++"
    if suffix in {".cs", ".csproj"}:
        return "C#"
    if suffix in {".fs", ".fsproj"}:
        return "F#"
    if suffix in {".vb", ".vbproj"}:
        return "VB.NET"
    if suffix == ".sln":
        return ".NET Solution"
    return "Unknown"


def is_dotnet_file(path: str) -> bool:
    return Path(path).suffix.lower() in DOTNET_SUFFIXES


def _walk_files(root_path: Path) -> Iterable[Path]:
    for current_root, dirs, files in os.walk(root_path):
        dirs[:] = [name for name in dirs if name not in IGNORED_DIRS]
        for filename in files:
            yield Path(current_root) / filename


def _parse_csproj_dependencies(project_path: Path) -> Dict[str, List[str]]:
    package_refs: List[str] = []
    project_refs: List[str] = []
    metadata: Dict[str, List[str] | str] = {
        "target_frameworks": [],
        "output_type": "",
        "nullable": "",
        "implicit_usings": "",
        "lang_version": "",
    }
    try:
        tree = ET.parse(project_path)
        root = tree.getroot()
        for element in root.iter():
            tag = element.tag.split("}")[-1]
            include = element.attrib.get("Include") or element.attrib.get("Update")
            if tag == "PackageReference" and include:
                package_refs.append(include)
            elif tag == "ProjectReference" and include:
                project_refs.append(include)
            elif tag in {"TargetFramework", "TargetFrameworks"}:
                values = [part.strip() for part in (element.text or "").split(";") if part.strip()]
                if values:
                    metadata["target_frameworks"].extend(values)
            elif tag == "OutputType":
                metadata["output_type"] = (element.text or "").strip()
            elif tag == "Nullable":
                metadata["nullable"] = (element.text or "").strip()
            elif tag == "ImplicitUsings":
                metadata["implicit_usings"] = (element.text or "").strip()
            elif tag == "LangVersion":
                metadata["lang_version"] = (element.text or "").strip()
    except ET.ParseError as exc:
        # malformed csproj
        import warnings

        warnings.warn(f"Failed to parse csproj {project_path}: {exc}")
    except Exception as exc:
        import warnings

        warnings.warn(f"Unexpected error parsing csproj {project_path}: {exc}")
    return {"packages": package_refs, "projects": project_refs, "metadata": metadata}


def _parse_source_imports(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        import warnings

        warnings.warn(f"Failed to read source for imports {path}: {exc}")
        return []

    imports: List[str] = []
    if path.suffix.lower() == ".cs":
        for match in re.findall(r"^\s*using\s+([A-Za-z0-9_.]+)\s*;", text, flags=re.MULTILINE):
            imports.append(match)
        for match in re.findall(r"^\s*global\s+using\s+([A-Za-z0-9_.]+)\s*;", text, flags=re.MULTILINE):
            imports.append(match)
    return imports


def _parse_source_references(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        import warnings

        warnings.warn(f"Failed to read source for references {path}: {exc}")
        return []

    references: List[str] = []
    if path.suffix.lower() == ".cs":
        if re.search(r"\bstatic\s+void\s+Main\s*\(", text):
            references.append("entrypoint:Main")
        if re.search(r"\bConsole\.Write(Line)?\s*\(", text):
            references.append("console-io")
        if re.search(r"\bawait\s+", text):
            references.append("async-await")
        if re.search(r"\bTask\b", text):
            references.append("task-based")
    return references


def _default_target_file(files: List[FileRecord]) -> Optional[FileRecord]:
    priority_names = ("program.cs", "main.cs", "startup.cs")
    for name in priority_names:
        for record in files:
            if record.relative_path.lower().endswith(name):
                return record

    for record in files:
        if record.language == "C#" and "entrypoint:Main" in record.references:
            return record

    for record in files:
        if record.language == "C#":
            return record

    for record in files:
        if record.language == "Python":
            return record

    return files[0] if files else None


def scan_project(root_path: str) -> Dict[str, object]:
    root = Path(root_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")

    files: List[FileRecord] = []
    csproj_files: List[str] = []
    sln_files: List[str] = []
    dotnet_projects: List[Dict[str, object]] = []
    total_bytes = 0
    entrypoint_files = 0
    code_file_count = 0

    for file_path in _walk_files(root):
        try:
            size_bytes = file_path.stat().st_size
        except OSError:
            size_bytes = 0
        total_bytes += size_bytes

        relative_path = str(file_path.relative_to(root)).replace("\\", "/")
        suffix = file_path.suffix.lower()
        language = detect_language(relative_path)
        imports = _parse_source_imports(file_path) if suffix in {".cs"} else []
        references = _parse_source_references(file_path) if suffix in {".cs"} else []
        if suffix in {".cs", ".fs", ".vb", ".cpp", ".cc", ".cxx", ".hpp", ".h"}:
            code_file_count += 1
        if "entrypoint:Main" in references:
            entrypoint_files += 1

        if suffix == ".csproj":
            csproj_files.append(relative_path)
            dep_info = _parse_csproj_dependencies(file_path)
            references.extend([f"package:{name}" for name in dep_info["packages"]])
            references.extend([f"project:{name}" for name in dep_info["projects"]])
            metadata = dep_info.get("metadata", {})
            dotnet_projects.append(
                {
                    "path": relative_path,
                    "package_refs": dep_info.get("packages", []),
                    "project_refs": dep_info.get("projects", []),
                    "target_frameworks": metadata.get("target_frameworks", []),
                    "output_type": metadata.get("output_type", ""),
                    "nullable": metadata.get("nullable", ""),
                    "implicit_usings": metadata.get("implicit_usings", ""),
                    "lang_version": metadata.get("lang_version", ""),
                }
            )
        elif suffix == ".sln":
            sln_files.append(relative_path)

        files.append(
            FileRecord(
                path=str(file_path),
                relative_path=relative_path,
                suffix=suffix,
                language=language,
                size_bytes=size_bytes,
                imports=imports,
                references=references,
            )
        )

    dotnet_files = [record for record in files if is_dotnet_file(record.path)]
    target = _default_target_file(files)
    project_type = "dotnet" if csproj_files or sln_files else "mixed"
    if not files:
        project_type = "empty"

    language_counts: Dict[str, int] = {}
    for record in files:
        language_counts[record.language] = language_counts.get(record.language, 0) + 1

    dependency_count = sum(
        len(project["package_refs"]) + len(project["project_refs"]) for project in dotnet_projects
    )
    average_file_size = total_bytes / len(files) if files else 0.0
    largest_files = sorted(files, key=lambda record: record.size_bytes, reverse=True)[:10]
    project_health_score = min(
        100,
        int(
            20
            + min(len(files), 100) * 0.35
            + min(code_file_count, 80) * 0.5
            + min(dependency_count, 50) * 1.2
            + min(entrypoint_files, 4) * 4
            - (25 if not csproj_files and not sln_files else 0)
        ),
    )

    dotnet_summary = {
        "solution_files": sln_files,
        "project_files": csproj_files,
        "candidate_files": [record.relative_path for record in files if record.language == "C#"],
        "project_metadata": dotnet_projects,
        "dependency_count": dependency_count,
        "entrypoint_files": entrypoint_files,
    }

    return {
        "root_path": str(root),
        "project_type": project_type,
        "file_count": len(files),
        "dotnet_file_count": len(dotnet_files),
        "language_counts": language_counts,
        "total_bytes": total_bytes,
        "average_file_size": average_file_size,
        "largest_files": [
            {"relative_path": record.relative_path, "size_bytes": record.size_bytes, "language": record.language}
            for record in largest_files
        ],
        "project_health_score": project_health_score,
        "dotnet_summary": dotnet_summary,
        "files": [record.__dict__ for record in files],
        "default_target": target.__dict__ if target else None,
    }


def clone_git_repo(repo_url: str, timeout_s: int = 120) -> str:
    repo_url = repo_url.strip()
    if not repo_url:
        raise ValueError("Repository URL is required")

    git_executable = shutil.which("git")
    if not git_executable:
        raise RuntimeError("Git is not installed or not available on PATH")

    target_dir = tempfile.mkdtemp(prefix="ecologic_repo_")
    result = subprocess.run(
        [git_executable, "clone", "--depth", "1", repo_url, target_dir],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "git clone failed").strip())
    return target_dir


def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")
