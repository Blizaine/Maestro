"""Best-effort, non-enforcing request field hints from public Maestro source.

This module parses installed ``launch.py`` and the single allowlisted
``services/director_pipeline.py`` file with :mod:`ast`. It never imports or
executes application code and never opens settings, docs, model metadata, or
private configuration files. OpenAPI remains authoritative when it publishes a
schema; these hints help callers discover fields for native ``Request.json()``
handlers that omit one.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import threading
from typing import Any


_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_SOURCE_BYTES = 20 * 1024 * 1024
_MAX_AST_NODES = 700_000
_MAX_HINT_FIELDS = 96
_MAX_HELPER_DEPTH = 5
_MAX_HELPER_FUNCTIONS = 16
_MAX_HELPER_CALLS = 96
_CACHE_LIMIT = 4
_ALLOWED_SOURCE_FILES = ("launch.py", "services/director_pipeline.py")
_CROSS_SOURCE_HELPERS = {
    ("launch.py", "start_pipeline"): "services/director_pipeline.py",
}
_PRIVATE_SOURCE_FIELDS = frozenset({"generated_reference_image_filename"})
_SENSITIVE_NAME = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|token|credential|private[_-]?key|"
    r"otp|passcode|access[_-]?code|activation[_-]?code)",
    re.I,
)
_CONTENT_NAME = re.compile(
    r"(?:prompt|negative|text|query|description|message|instruction|content|lyrics|"
    r"script|story|scene|caption|note|path|filename|url|reference|system)",
    re.I,
)
_TYPE_NAMES = {
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "str": "string",
    "list": "array",
    "tuple": "array",
    "dict": "object",
    "set": "array",
}
_CACHE: OrderedDict[tuple[str, str], "ContractIndex"] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _route_key(method: str, path: str) -> tuple[str, str]:
    return str(method or "").upper(), str(path or "").rstrip("/") or "/"


def _annotation_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _request_parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    if function.args.vararg is not None:
        arguments.append(function.args.vararg)
    if function.args.kwarg is not None:
        arguments.append(function.args.kwarg)
    for argument in arguments:
        if argument.arg.casefold() == "request" or _annotation_name(argument.annotation) == "Request":
            names.add(argument.arg)
    return names


def _is_request_json(node: ast.AST, request_names: set[str]) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    return (
        node.func.attr == "json"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in request_names
    )


class _ScopeVisitor(ast.NodeVisitor):
    """Visit a function body without walking into nested scopes."""

    def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef):
        self.root = root

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if node is self.root:
            self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda):
        return

    def visit_ClassDef(self, node: ast.ClassDef):
        return


def _literal_type(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple, set)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def _safe_default(name: str, value: Any) -> tuple[bool, Any]:
    """Return a safe scalar default, never source text or sensitive values."""

    if _SENSITIVE_NAME.search(name):
        return False, None
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return True, value
    if isinstance(value, float):
        return (math.isfinite(value), value if math.isfinite(value) else None)
    if isinstance(value, str):
        if not value:
            return True, ""
        if (
            len(value) <= 32
            and "\n" not in value
            and "\r" not in value
            and not _CONTENT_NAME.search(name)
            and not re.search(r"(?:https?://|[A-Za-z]:[\\/])", value, re.I)
        ):
            return True, value
    return False, None


def _string_constant(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _safe_field_name(name: str) -> bool:
    return bool(
        name
        and not name.startswith("_")
        and len(name) <= 64
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        and not _SENSITIVE_NAME.search(name)
        and name not in _PRIVATE_SOURCE_FIELDS
    )


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(_target_names(item))
        return names
    return set()


def _contains_request_json(node: ast.AST, request_names: set[str]) -> bool:
    return any(_is_request_json(child, request_names) for child in ast.walk(node))


def _decorator_routes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        owner = decorator.func.value
        if not isinstance(owner, ast.Name) or owner.id != "api":
            continue
        route = decorator.args[0].value if decorator.args and isinstance(decorator.args[0], ast.Constant) else None
        if not isinstance(route, str) or not route.startswith("/api/v1/"):
            continue
        method_name = decorator.func.attr.casefold()
        if method_name in {"get", "post", "put", "patch", "delete"}:
            routes.append((method_name.upper(), route))
            continue
        if method_name not in {"route", "api_route"}:
            continue
        methods: set[str] = set()
        for keyword in decorator.keywords:
            if keyword.arg == "methods" and isinstance(keyword.value, (ast.List, ast.Tuple, ast.Set)):
                methods.update(
                    item.value.upper()
                    for item in keyword.value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
        routes.extend((method, route) for method in sorted(methods & {"GET", "POST", "PUT", "PATCH", "DELETE"}))
    return routes


def _module_functions(
    tree: ast.Module,
) -> tuple[dict[str, ast.FunctionDef | ast.AsyncFunctionDef], dict[tuple[str, str], list[ast.FunctionDef | ast.AsyncFunctionDef]]]:
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    routes: dict[tuple[str, str], list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        functions[node.name] = node
        for method, path in _decorator_routes(node):
            routes.setdefault(_route_key(method, path), []).append(node)
    return functions, routes


class _FieldVisitor(_ScopeVisitor):
    def __init__(
        self,
        root: ast.FunctionDef | ast.AsyncFunctionDef,
        aliases: set[str],
        request_names: set[str] | None = None,
    ):
        super().__init__(root)
        self.aliases = aliases
        self.request_names = request_names or set()
        self.fields: dict[str, dict[str, Any]] = {}
        self.helper_calls: list[tuple[str, ast.Call, tuple[ast.AST, ...]]] = []
        self.handles_key_error = False

    def _is_body_expression(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return True
        return isinstance(node, ast.Await) and _is_request_json(node.value, self.request_names)

    def _field(self, name: str) -> dict[str, Any] | None:
        if not _safe_field_name(name):
            return None
        return self.fields.setdefault(
            name,
            {"get_seen": False, "subscript_seen": False, "membership_seen": False, "types": [], "defaults": []},
        )

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "pop", "setdefault"}:
            if self._is_body_expression(node.func.value) and node.args:
                name = _string_constant(node.args[0])
                record = self._field(name) if name is not None else None
                if record is not None:
                    record["get_seen"] = record["get_seen"] or (
                        node.func.attr != "pop" or len(node.args) >= 2
                    )
                    if len(node.args) >= 2:
                        try:
                            value = ast.literal_eval(node.args[1])
                        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                            value = _MISSING
                        if value is not _MISSING:
                            value_type = _literal_type(value)
                            if value_type is not None:
                                record["types"].append(value_type)
                            safe, public = _safe_default(name, value)
                            record["defaults"].append((safe, public, value_type))
        if isinstance(node.func, ast.Name):
            self.helper_calls.append((node.func.id, node, tuple(node.args)))
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
            and node.func.attr == "to_thread"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            # FastAPI handlers commonly dispatch the blocking Director helper
            # through asyncio.to_thread(start_pipeline, body).
            self.helper_calls.append((node.args[0].id, node, tuple(node.args[1:])))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, ast.Load) and self._is_body_expression(node.value):
            name = _string_constant(node.slice)
            record = self._field(name) if name is not None else None
            if record is not None:
                record["subscript_seen"] = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        for index, operator in enumerate(node.ops):
            if not isinstance(operator, (ast.In, ast.NotIn)):
                continue
            left = node.left if index == 0 else node.comparators[index - 1]
            right = node.comparators[index]
            if self._is_body_expression(left):
                name = _string_constant(right)
            elif self._is_body_expression(right):
                name = _string_constant(left)
            else:
                name = None
            record = self._field(name) if name is not None else None
            if record is not None:
                record["membership_seen"] = True
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        if any(_annotation_name(item) in {"KeyError", "Exception", "BaseException"} for item in types if item is not None):
            self.handles_key_error = True
        self.generic_visit(node)

_MISSING = object()


def _body_aliases(function: ast.FunctionDef | ast.AsyncFunctionDef, request_names: set[str]) -> set[str]:
    aliases: set[str] = set()
    assignments: list[tuple[set[str], ast.AST]] = []
    class AssignmentVisitor(_ScopeVisitor):
        def __init__(self, root):
            super().__init__(root)
            self.assignments: list[tuple[set[str], ast.AST]] = []

        def visit_Assign(self, node: ast.Assign):
            self.assignments.append((set().union(*(_target_names(target) for target in node.targets)), node.value))
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if node.value is not None:
                self.assignments.append((_target_names(node.target), node.value))
            self.generic_visit(node)

        def visit_NamedExpr(self, node: ast.NamedExpr):
            self.assignments.append((_target_names(node.target), node.value))
            self.generic_visit(node)

    assignment_visitor = AssignmentVisitor(function)
    assignment_visitor.visit(function)
    assignments = assignment_visitor.assignments
    changed = True
    while changed:
        changed = False
        for targets, value in assignments:
            seeded = _contains_request_json(value, request_names)
            identity = isinstance(value, ast.Name) and value.id in aliases
            if seeded or identity:
                for target in targets:
                    if target not in aliases:
                        aliases.add(target)
                        changed = True
    return aliases


def _contains_request_json_in_scope(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    request_names: set[str],
) -> bool:
    class RequestJsonVisitor(_ScopeVisitor):
        def __init__(self, root):
            super().__init__(root)
            self.found = False

        def visit_Call(self, node: ast.Call):
            if _is_request_json(node, request_names):
                self.found = True
            self.generic_visit(node)

    visitor = RequestJsonVisitor(function)
    visitor.visit(function)
    return visitor.found


def _helper_aliases(
    call: ast.Call,
    arguments: tuple[ast.AST, ...],
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: set[str],
    request_names: set[str],
) -> set[str]:
    parameters = [*function.args.posonlyargs, *function.args.args]
    keyword_parameters = [*parameters, *function.args.kwonlyargs]
    mapped: set[str] = set()
    for index, argument in enumerate(arguments):
        if index >= len(parameters):
            break
        if isinstance(argument, ast.Name) and argument.id in aliases or _contains_request_json(argument, request_names):
            mapped.add(parameters[index].arg)
    parameter_positions = {argument.arg for argument in keyword_parameters}
    for keyword in call.keywords:
        if keyword.arg in parameter_positions and (
            isinstance(keyword.value, ast.Name) and keyword.value.id in aliases
            or _contains_request_json(keyword.value, request_names)
        ):
            mapped.add(keyword.arg)
    return mapped


def _resolve_helper(
    source_name: str,
    helper_name: str,
    functions_by_source: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    local = functions_by_source.get(source_name, {}).get(helper_name)
    if local is not None:
        return source_name, local
    target_source = _CROSS_SOURCE_HELPERS.get((source_name, helper_name))
    target = functions_by_source.get(target_source or "", {}).get(helper_name)
    if target is not None and target_source is not None:
        return target_source, target
    return None


def _isinstance_hints(
    functions: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]],
) -> dict[str, set[str]]:
    hints: dict[str, set[str]] = {}
    for function, aliases in functions:
        class TypeVisitor(_ScopeVisitor):
            def visit_Call(self, node: ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "isinstance"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], (ast.Name, ast.Tuple))
                ):
                    candidate = node.args[0]
                    if (
                        isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Attribute)
                        and candidate.func.attr == "get"
                        and isinstance(candidate.func.value, ast.Name)
                        and candidate.func.value.id in aliases
                        and candidate.args
                    ):
                        field_name = _string_constant(candidate.args[0])
                        if field_name and _safe_field_name(field_name):
                            type_nodes = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
                            values = {
                                _TYPE_NAMES[item.id]
                                for item in type_nodes
                                if isinstance(item, ast.Name) and item.id in _TYPE_NAMES
                            }
                            if len(values) == 1:
                                hints.setdefault(field_name, set()).update(values)
                self.generic_visit(node)

        visitor = TypeVisitor(function)
        visitor.visit(function)
    return hints


def _collect_fields(
    root: ast.FunctionDef | ast.AsyncFunctionDef,
    root_source: str,
    functions_by_source: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
) -> tuple[list[dict[str, Any]], bool, bool, list[str]]:
    root_request_names = _request_parameters(root)
    root_aliases = _body_aliases(root, root_request_names)
    root_has_request_json = _contains_request_json_in_scope(root, root_request_names)
    queue: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, set[str], int]] = [
        (root_source, root, root_aliases, 0)
    ]
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    scopes: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, set[str]]] = []
    fields: dict[str, dict[str, Any]] = {}
    saw_request_json = bool(root_aliases) or root_has_request_json
    truncated = False
    used_sources = {root_source}
    helper_call_count = 0

    while queue:
        source_name, function, aliases, depth = queue.pop(0)
        if not aliases and not (function is root and root_has_request_json):
            continue
        identity = (source_name, function.name, tuple(sorted(aliases)))
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen) > _MAX_HELPER_FUNCTIONS:
            truncated = True
            break
        scopes.append((function, aliases))
        request_names = root_request_names if function is root else _request_parameters(function)
        visitor = _FieldVisitor(function, aliases, request_names)
        visitor.visit(function)
        for name, item in visitor.fields.items():
            target = fields.setdefault(
                name,
                {"get_seen": False, "subscript_seen": False, "membership_seen": False, "types": [], "defaults": []},
            )
            for key in ("get_seen", "subscript_seen", "membership_seen"):
                target[key] = target[key] or item[key]
            target["types"].extend(item["types"])
            target["defaults"].extend(item["defaults"])
            if visitor.handles_key_error:
                target["key_error_guard"] = True
        for helper_name, call, arguments in visitor.helper_calls:
            resolved = _resolve_helper(source_name, helper_name, functions_by_source)
            if resolved is None:
                continue
            helper_call_count += 1
            if helper_call_count > _MAX_HELPER_CALLS:
                truncated = True
                break
            helper_source, helper = resolved
            helper_aliases = _helper_aliases(call, arguments, helper, aliases, request_names)
            if helper_aliases:
                if depth >= _MAX_HELPER_DEPTH:
                    truncated = True
                else:
                    used_sources.add(helper_source)
                    queue.append((helper_source, helper, helper_aliases, depth + 1))

    type_hints = _isinstance_hints(scopes)
    result: list[dict[str, Any]] = []
    for name, record in fields.items():
        types = set(record["types"]) | type_hints.get(name, set())
        inferred_type = next(iter(types)) if len(types) == 1 else None
        if inferred_type == "null":
            inferred_type = None
        defaults = record["defaults"]
        known_defaults = [item for item in defaults if item[0]]
        public_default = _MISSING
        if defaults and len(known_defaults) == len(defaults):
            serialized = {(type(value).__name__, repr(value)) for _, value, _ in known_defaults}
            if len(serialized) == 1:
                public_default = known_defaults[0][1]
        if inferred_type is None and public_default is not _MISSING:
            default_type = _literal_type(public_default)
            if default_type != "null":
                inferred_type = default_type
        item: dict[str, Any] = {"name": name}
        if inferred_type is not None:
            item["type"] = inferred_type
        item["required"] = bool(
            record["subscript_seen"]
            and not record["get_seen"]
            and not record["membership_seen"]
            and not record.get("key_error_guard")
        )
        if public_default is not _MISSING:
            item["default"] = public_default
        result.append(item)
    if len(result) > _MAX_HINT_FIELDS:
        result = result[:_MAX_HINT_FIELDS]
        truncated = True
    return result, saw_request_json, truncated, sorted(used_sources)


@dataclass(frozen=True)
class ContractIndex:
    """Parsed source field hints for one immutable source hash."""

    source_hash: str | None
    status: str
    functions: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]
    routes: dict[tuple[str, str], tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]]

    @classmethod
    def unavailable(cls, status: str = "unavailable") -> "ContractIndex":
        return cls(None, status, {}, {})

    def describe(self, method: str, path: str) -> dict[str, Any]:
        base = {
            "source": "launch.py",
            "sources": ["launch.py"],
            "validation": "Hints only; native validation is authoritative.",
            "fields": [],
            "truncated": False,
        }
        if self.status != "available":
            return {**base, "status": self.status}
        handlers = self.routes.get(_route_key(method, path), ())
        if not handlers:
            return {**base, "status": "unavailable"}
        if len(handlers) != 1:
            return {**base, "status": "ambiguous"}
        root_source, handler = handlers[0]
        fields, saw_request_json, truncated, sources = _collect_fields(
            handler, root_source, self.functions,
        )
        if not saw_request_json:
            return {**base, "status": "no_request_json", "sources": sources}
        return {
            **base,
            "status": "available",
            "fields": fields,
            "truncated": truncated,
            "sources": sources,
        }


def _parse_sources(sources: dict[str, bytes]) -> ContractIndex:
    total_bytes = sum(len(value) for value in sources.values())
    if total_bytes > _MAX_TOTAL_SOURCE_BYTES or "launch.py" not in sources:
        return ContractIndex.unavailable("unavailable")
    functions_by_source: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    routes: dict[
        tuple[str, str],
        list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    ] = {}
    total_nodes = 0
    hash_builder = hashlib.sha256()
    for source_name in _ALLOWED_SOURCE_FILES:
        source = sources.get(source_name)
        if source is None:
            continue
        try:
            tree = ast.parse(source.decode("utf-8"), filename=source_name, mode="exec")
        except (UnicodeDecodeError, SyntaxError, ValueError, MemoryError, RecursionError):
            if source_name == "launch.py":
                return ContractIndex.unavailable("unavailable")
            continue
        total_nodes += sum(1 for _ in ast.walk(tree))
        if total_nodes > _MAX_AST_NODES:
            return ContractIndex.unavailable("unavailable")
        module_functions, module_routes = _module_functions(tree)
        functions_by_source[source_name] = module_functions
        hash_builder.update(source_name.encode("utf-8"))
        hash_builder.update(b"\0")
        hash_builder.update(hashlib.sha256(source).digest())
        if source_name == "launch.py":
            for key, handlers in module_routes.items():
                routes.setdefault(key, []).extend((source_name, handler) for handler in handlers)
    return ContractIndex(
        hash_builder.hexdigest(),
        "available",
        functions_by_source,
        {key: tuple(value) for key, value in routes.items()},
    )


def load_contract_index(app_root: str | os.PathLike[str] | Path) -> ContractIndex:
    """Read the fixed allowlist under ``app_root`` once per catalogue revision.

    Only ``launch.py`` and ``services/director_pipeline.py`` may be opened.
    Parsed indexes are cached by the combined source-file SHA-256 digest. A
    missing, oversized, invalid, or unreadable launch source disables hints;
    the optional Director source may be absent without disabling launch hints.
    """

    sources: dict[str, bytes] = {}
    try:
        root = Path(app_root).resolve(strict=True)
        root_launch = root / "launch.py"
        for relative_name in _ALLOWED_SOURCE_FILES:
            path = root / Path(relative_name)
            try:
                if path.is_symlink():
                    if relative_name == "launch.py":
                        return ContractIndex.unavailable()
                    continue
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(root)
                if resolved_path.name.casefold() != Path(relative_name).name.casefold():
                    if relative_name == "launch.py":
                        return ContractIndex.unavailable()
                    continue
                stat = os.stat(resolved_path)
                if stat.st_size < 1 or stat.st_size > _MAX_SOURCE_BYTES:
                    if relative_name == "launch.py":
                        return ContractIndex.unavailable()
                    continue
                source = resolved_path.read_bytes()
                if len(source) != stat.st_size:
                    if relative_name == "launch.py":
                        return ContractIndex.unavailable()
                    continue
                sources[relative_name] = source
            except (OSError, RuntimeError, ValueError):
                if relative_name == "launch.py":
                    return ContractIndex.unavailable()
                continue
        if "launch.py" not in sources:
            return ContractIndex.unavailable()
        source_hash = hashlib.sha256(
            b"".join(
                name.encode("utf-8") + b"\0" + hashlib.sha256(sources[name]).digest()
                for name in _ALLOWED_SOURCE_FILES if name in sources
            )
        ).hexdigest()
        resolved = str(root_launch)
    except (OSError, RuntimeError, ValueError, TypeError):
        return ContractIndex.unavailable()
    key = (resolved, source_hash)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    parsed = _parse_sources(sources)
    with _CACHE_LOCK:
        _CACHE[key] = parsed
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return parsed


__all__ = ["ContractIndex", "load_contract_index"]
