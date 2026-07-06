"""TerminalTool - 工作区工具和命令执行。

- read_file: 按行号范围读取工作区文件
- glob_files: 按 glob 模式查找文件
- grep_code: 搜索文件内容
- write_file: 创建或覆盖工作区文件
- edit_file: 替换已有工作区文件中的文本
- bash: 执行受限 shell 命令
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess

from ..base import Tool, ToolParameter, tool_action


class TerminalTool(Tool):
    """工作区工具集合。

    ``expandable=False`` 保持原有单工具形态：
    ``terminal.run({"command": "..."})``.

    ``expandable=True`` 允许 ``ToolRegistry`` 为 LLM 函数调用暴露独立工具：
    ``read_file``、``glob_files``、``grep_code``、``write_file``、
    ``edit_file`` 和 ``bash``。
    """

    DEFAULT_EXCLUDED_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        ".next",
        ".turbo",
    }

    # 受限 bash 有意保持保守策略。读取、搜索和编辑应优先走上面的结构化动作；
    # bash 只用于测试和环境检查，不作为通用文件访问入口。
    ALLOWED_COMMANDS = {
        "pytest",
        "ruff",
        "mypy",
        "python",
        "python3",
        "pip",
        "uv",
        "poetry",
        "git",
        "npm",
        "pnpm",
        "yarn",
        "node",
        "rg",
        "grep",
        "findstr",
        "find",
        "ls",
        "dir",
        "tree",
        "pwd",
        "echo",
        "where",
        "which",
        "type",
        "cat",
        "head",
        "tail",
        "wc",
    }

    SHELL_OPERATORS = ("&&", "||", ";", "|", ">", "<", "`", "$(")

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 30,
        max_output_size: int = 10 * 1024 * 1024,
        allow_cd: bool = True,
        os_type: str = "auto",
        expandable: bool = False,
        restricted_bash: bool = True,
        allowed_commands: Optional[List[str]] = None,
        excluded_dirs: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        super().__init__(
            name="terminal",
            description=(
                "工作区工具集合 - 读取/搜索/写入/编辑文件，并在受限模式下执行命令"
            ),
            expandable=expandable,
        )

        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_cd = allow_cd
        self.restricted_bash = restricted_bash
        self.allowed_commands = set(allowed_commands or self.ALLOWED_COMMANDS)
        self.excluded_dirs = set(excluded_dirs or self.DEFAULT_EXCLUDED_DIRS)
        self.encoding = encoding

        system = platform.system().lower()
        if os_type == "auto":
            self.os_type = {"windows": "windows", "darwin": "mac"}.get(system, "linux")
        else:
            self.os_type = os_type.lower()

        self.current_dir = self.workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行未展开形态的 terminal 工具。

        向后兼容：
        - ``{"command": "pytest"}`` 会按 ``bash`` 执行。
        - ``{"action": "read_file", "path": "..."}`` 会分发到结构化动作。
        """
        parameters = parameters or {}
        action = str(parameters.get("action") or "").strip().lower()

        if not action and "command" in parameters:
            return self._bash(command=str(parameters.get("command") or ""))

        if action == "bash":
            return self._bash(command=str(parameters.get("command") or ""))
        if action == "read_file":
            return self._read_file(
                path=str(parameters.get("path") or ""),
                start_line=self._as_int(parameters.get("start_line"), 1),
                end_line=self._as_int(parameters.get("end_line"), 0),
            )
        if action == "glob_files":
            return self._glob_files(
                pattern=str(parameters.get("pattern") or "**/*"),
                limit=self._as_int(parameters.get("limit"), 100),
                include_dirs=self._as_bool(parameters.get("include_dirs"), False),
            )
        if action == "grep_code":
            return self._grep_code(
                query=str(parameters.get("query") or parameters.get("pattern") or ""),
                include=str(parameters.get("include") or "**/*"),
                case_sensitive=self._as_bool(parameters.get("case_sensitive"), False),
                regex=self._as_bool(parameters.get("regex"), False),
                limit=self._as_int(parameters.get("limit"), 50),
            )
        if action == "write_file":
            return self._write_file(
                path=str(parameters.get("path") or ""),
                content=str(parameters.get("content") or ""),
                overwrite=self._as_bool(parameters.get("overwrite"), False),
                create_dirs=self._as_bool(parameters.get("create_dirs"), True),
            )
        if action == "edit_file":
            return self._edit_file(
                path=str(parameters.get("path") or ""),
                old_text=str(parameters.get("old_text") or ""),
                new_text=str(parameters.get("new_text") or ""),
                replace_all=self._as_bool(parameters.get("replace_all"), False),
            )

        return self._error(
            "不支持的操作",
            supported_actions=[
                "read_file",
                "glob_files",
                "grep_code",
                "write_file",
                "edit_file",
                "bash",
            ],
        )

    def get_parameters(self) -> List[ToolParameter]:
        """向后兼容的单 terminal 工具参数定义。"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "可选操作：read_file/glob_files/grep_code/write_file/edit_file/bash。"
                    "省略 action 且提供 command 时，按 bash 执行。"
                ),
                required=False,
            ),
            ToolParameter(name="command", type="string", description="bash 命令", required=False),
            ToolParameter(name="path", type="string", description="工作区内文件路径", required=False),
            ToolParameter(name="pattern", type="string", description="glob 模式或搜索模式", required=False),
            ToolParameter(name="query", type="string", description="grep 搜索文本或正则", required=False),
            ToolParameter(name="content", type="string", description="write_file 写入内容", required=False),
            ToolParameter(name="old_text", type="string", description="edit_file 要替换的原文本", required=False),
            ToolParameter(name="new_text", type="string", description="edit_file 替换后的文本", required=False),
            ToolParameter(name="start_line", type="integer", description="read_file 起始行，1 基", required=False),
            ToolParameter(name="end_line", type="integer", description="read_file 结束行，0 表示最多读取 500 行", required=False),
            ToolParameter(name="limit", type="integer", description="最大返回条数", required=False),
            ToolParameter(name="case_sensitive", type="boolean", description="grep 是否大小写敏感", required=False),
            ToolParameter(name="regex", type="boolean", description="grep query 是否按正则解释", required=False),
            ToolParameter(name="overwrite", type="boolean", description="write_file 是否允许覆盖", required=False),
            ToolParameter(name="replace_all", type="boolean", description="edit_file 是否替换全部匹配", required=False),
            ToolParameter(name="create_dirs", type="boolean", description="write_file 是否自动创建父目录", required=False),
            ToolParameter(name="include_dirs", type="boolean", description="glob_files 是否包含目录", required=False),
        ]

    @tool_action("read_file", "读取工作区内文件内容，支持起止行号")
    def _read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        """读取工作区内文件。

        参数：
            path: 工作区内文件路径
            start_line: 起始行号，1 基
            end_line: 结束行号；0 表示从 start_line 开始最多读取 500 行
        """
        try:
            file_path = self._resolve_path(path, must_exist=True)
            if not file_path.is_file():
                return self._error("路径不是文件", path=self._rel(file_path))

            start = max(1, int(start_line or 1))
            end = int(end_line or 0)
            if end <= 0:
                end = start + 499
            if end < start:
                return self._error("end_line 不能小于 start_line")

            selected: List[str] = []
            total_lines = 0
            with file_path.open("r", encoding=self.encoding, errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    total_lines = line_no
                    if start <= line_no <= end:
                        selected.append(f"{line_no}: {line.rstrip()}")

            content = "\n".join(selected)
            content, truncated = self._truncate(content)
            return self._ok(
                path=self._rel(file_path),
                start_line=start,
                end_line=min(end, total_lines) if total_lines else end,
                total_lines=total_lines,
                truncated=truncated,
                content=content,
            )
        except Exception as exc:
            return self._error(str(exc))

    @tool_action("glob_files", "按 glob 模式查找工作区文件")
    def _glob_files(self, pattern: str = "**/*", limit: int = 100, include_dirs: bool = False) -> str:
        """按 glob 模式查找文件。

        参数：
            pattern: glob 模式，例如 "**/*.py"、"agent/**/*.py"
            limit: 最多返回多少条结果
            include_dirs: 是否包含目录
        """
        try:
            pattern = (pattern or "**/*").strip()
            max_items = max(1, int(limit or 100))
            matches = [
                {
                    "path": self._rel(item),
                    "type": "dir" if item.is_dir() else "file",
                }
                for item in self._iter_workspace_paths(pattern, include_dirs=include_dirs)
            ]

            matches.sort(key=lambda entry: entry["path"])
            truncated = len(matches) > max_items
            return self._ok(
                pattern=pattern,
                root=self._rel(self.current_dir),
                count=len(matches),
                truncated=truncated,
                matches=matches[:max_items],
            )
        except Exception as exc:
            return self._error(str(exc))

    @tool_action("grep_code", "搜索工作区文件内容并返回文件路径、行号和匹配行")
    def _grep_code(
        self,
        query: str,
        include: str = "**/*",
        case_sensitive: bool = False,
        regex: bool = False,
        limit: int = 50,
    ) -> str:
        """搜索文件内容。

        参数：
            query: 搜索文本；regex=True 时按正则表达式解释
            include: 限制搜索文件的 glob 模式，例如 "**/*.py"
            case_sensitive: 是否大小写敏感
            regex: 是否启用正则搜索
            limit: 最多返回多少条匹配
        """
        try:
            query = (query or "").strip()
            if not query:
                return self._error("query 不能为空")
            include = (include or "**/*").strip()
            max_items = max(1, int(limit or 50))

            flags = 0 if case_sensitive else re.IGNORECASE
            expr = re.compile(query if regex else re.escape(query), flags)

            matches = []
            searched_files = 0
            truncated = False

            for file_path in self._iter_workspace_paths(include):
                if not file_path.is_file():
                    continue
                try:
                    with file_path.open("rb") as handle:
                        if b"\0" in handle.read(1024):
                            continue
                except OSError:
                    continue

                searched_files += 1
                try:
                    with file_path.open("r", encoding=self.encoding, errors="replace") as handle:
                        for line_no, line in enumerate(handle, start=1):
                            if expr.search(line):
                                text = line.rstrip()
                                if len(text) > 500:
                                    text = text[:500] + "...[truncated]"
                                matches.append(
                                    {
                                        "path": self._rel(file_path),
                                        "line": line_no,
                                        "text": text,
                                    }
                                )
                                if len(matches) >= max_items:
                                    truncated = True
                                    raise StopIteration
                except StopIteration:
                    break
                except OSError:
                    continue

            return self._ok(
                query=query,
                include=include,
                regex=regex,
                case_sensitive=case_sensitive,
                searched_files=searched_files,
                count=len(matches),
                truncated=truncated,
                matches=matches,
            )
        except re.error as exc:
            return self._error(f"正则表达式无效: {exc}")
        except Exception as exc:
            return self._error(str(exc))

    @tool_action("write_file", "在工作区内写入文件，可选择是否覆盖")
    def _write_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        create_dirs: bool = True,
    ) -> str:
        """写入工作区文件。

        参数：
            path: 工作区内文件路径
            content: 要写入的完整文件内容
            overwrite: 目标存在时是否允许覆盖
            create_dirs: 父目录不存在时是否自动创建
        """
        try:
            file_path = self._resolve_path(path, must_exist=False)
            existed = file_path.exists()
            if existed and file_path.is_dir():
                return self._error("目标路径是目录", path=self._rel(file_path))
            if existed and not overwrite:
                return self._error("文件已存在；如需覆盖请设置 overwrite=true", path=self._rel(file_path))
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            elif not file_path.parent.exists():
                return self._error("父目录不存在", parent=self._rel(file_path.parent))

            file_path.write_text(content or "", encoding=self.encoding)
            return self._ok(
                path=self._rel(file_path),
                bytes=file_path.stat().st_size,
                overwritten=existed and overwrite,
            )
        except Exception as exc:
            return self._error(str(exc))

    @tool_action("edit_file", "替换工作区内已有文件的指定文本")
    def _edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        """通过精确文本替换编辑已有工作区文件。

        参数：
            path: 工作区内已有文件路径
            old_text: 要替换的原始文本，必须精确匹配
            new_text: 替换后的文本
            replace_all: 是否替换全部匹配；默认只替换第一处
        """
        try:
            if old_text == "":
                return self._error("old_text 不能为空")

            file_path = self._resolve_path(path, must_exist=True)
            if not file_path.is_file():
                return self._error("路径不是文件", path=self._rel(file_path))

            content = file_path.read_text(encoding=self.encoding, errors="replace")
            occurrences = content.count(old_text)
            if occurrences == 0:
                return self._error("未找到 old_text", path=self._rel(file_path))

            replacements = occurrences if replace_all else 1
            updated = content.replace(old_text, new_text, replacements)
            file_path.write_text(updated, encoding=self.encoding)

            return self._ok(
                path=self._rel(file_path),
                replacements=replacements,
                remaining_occurrences=0 if replace_all else max(0, occurrences - 1),
                bytes=file_path.stat().st_size,
            )
        except Exception as exc:
            return self._error(str(exc))

    @tool_action("bash", "在当前工作目录执行受限命令，用于测试、构建和环境检查")
    def _bash(self, command: str) -> str:
        """在当前工作区目录执行 shell 命令。

        参数：
            command: 要执行的命令。受限模式下只允许白名单命令，且拒绝 shell 管道/重定向/链式操作。
        """
        command = (command or "").strip()
        if not command:
            return self._error("命令不能为空")

        try:
            parts = shlex.split(command, posix=self.os_type != "windows")
        except ValueError as exc:
            return self._error(f"命令解析失败: {exc}")
        if not parts:
            return self._error("命令不能为空")

        if parts[0].lower() == "cd":
            return self._handle_cd(parts)

        if self.restricted_bash:
            base_command = Path(parts[0]).name.lower()
            if base_command not in self.allowed_commands:
                return self._error(
                    "命令不在白名单中",
                    command=base_command,
                    allowed_commands=sorted(self.allowed_commands),
                )
            for operator in self.SHELL_OPERATORS:
                if operator in command:
                    return self._error(
                        "受限 bash 不允许 shell 管道、重定向或链式操作",
                        operator=operator,
                    )

        return self._execute_command(command)

    def _handle_cd(self, parts: List[str]) -> str:
        """处理持久化目录切换。"""
        if not self.allow_cd:
            return self._error("cd 命令已禁用")

        if len(parts) < 2:
            return self._ok(current_dir=self._rel(self.current_dir))

        target_dir = parts[1]
        if target_dir == "~":
            new_dir = self.workspace
        else:
            new_dir = self._resolve_path(target_dir, must_exist=True)

        if not new_dir.is_dir():
            return self._error("不是目录", path=self._rel(new_dir))

        self.current_dir = new_dir
        return self._ok(current_dir=self._rel(self.current_dir))

    def _execute_command(self, command: str) -> str:
        """执行命令并返回结构化观察结果。"""
        try:
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")

            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.current_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=env,
            )

            stdout, stdout_truncated = self._truncate(result.stdout or "")
            stderr, stderr_truncated = self._truncate(result.stderr or "")
            return self._ok(
                command=command,
                cwd=self._rel(self.current_dir),
                exit_code=result.returncode,
                command_status="succeeded" if result.returncode == 0 else "failed",
                stdout=stdout,
                stderr=stderr,
                truncated=stdout_truncated or stderr_truncated,
            )
        except subprocess.TimeoutExpired:
            return self._error(
                f"命令执行超时（超过 {self.timeout} 秒）",
                command=command,
                cwd=self._rel(self.current_dir),
            )
        except Exception as exc:
            return self._error(f"命令执行失败: {exc}", command=command, cwd=self._rel(self.current_dir))

    def _resolve_path(self, path: str, must_exist: bool = False) -> Path:
        raw = (path or "").strip()
        if not raw:
            raise ValueError("路径不能为空")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.current_dir / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"不允许访问工作目录外的路径: {resolved}") from exc
        if must_exist and not resolved.exists():
            raise ValueError(f"路径不存在: {self._rel(resolved)}")
        return resolved

    def _iter_workspace_paths(self, pattern: str, *, include_dirs: bool = False):
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            raise ValueError("glob/include 模式必须是工作区内相对模式")
        if ".." in pattern_path.parts:
            raise ValueError("glob/include 模式不允许包含 '..'")

        for item in self.current_dir.glob(pattern):
            try:
                item = item.resolve()
                rel = item.relative_to(self.workspace)
            except ValueError:
                continue
            if any(part in self.excluded_dirs for part in rel.parts):
                continue
            if item.is_dir() and not include_dirs:
                continue
            yield item

    def _rel(self, path: Path) -> str:
        try:
            if path == self.workspace:
                return "."
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_size:
            return text, False

        marker = "\n...[truncated middle]...\n"
        if self.max_output_size <= len(marker) + 2:
            return text[: self.max_output_size], True

        remaining = self.max_output_size - len(marker)
        head_size = max(1, remaining // 2)
        tail_size = max(1, remaining - head_size)
        return text[:head_size] + marker + text[-tail_size:], True

    def _ok(self, **payload: Any) -> str:
        return json.dumps({"status": "success", **payload}, ensure_ascii=False, indent=2)

    def _error(self, message: str, **payload: Any) -> str:
        return json.dumps({"status": "error", "error": message, **payload}, ensure_ascii=False, indent=2)

    def _as_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
