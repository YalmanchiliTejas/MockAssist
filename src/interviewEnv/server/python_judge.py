"""Build a self-contained oracle test program for LeetCode-style Python code."""

from __future__ import annotations

import ast
import html
import json
import re
from copy import deepcopy
from typing import Any


RESULT_MARKER = "__MOCKASSIST_TEST_RESULT__="


def build_test_program(candidate_code: str, problem: dict[str, Any]) -> str | None:
    reference_code = (problem.get("metadata") or {}).get("reference_code_python")
    design_examples = _extract_design_examples(problem.get("description", ""))
    if design_examples:
        class_name = _first_class_name(reference_code or "") or design_examples[0]["operations"][0]
        tests = _with_design_variants(design_examples, class_name, bool(reference_code))
        return _design_program(candidate_code, reference_code, class_name, tests)
    if not reference_code:
        return None
    method_name, annotations = _reference_signature(reference_code)
    if not method_name:
        return None
    examples = _extract_examples(problem.get("description", ""))
    if not examples:
        return None
    tests = _with_generated_variants(examples)
    unordered = "any order" in html.unescape(problem.get("description", "")).lower()
    return _program(
        candidate_code=candidate_code,
        reference_code=reference_code,
        method_name=method_name,
        annotations=annotations,
        tests=tests,
        unordered=unordered,
    )


def parse_test_report(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                value = json.loads(line[len(RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _reference_signature(code: str) -> tuple[str | None, dict[str, str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    annotations = {}
                    for arg in item.args.args:
                        if arg.arg == "self" or arg.annotation is None:
                            continue
                        annotations[arg.arg] = ast.unparse(arg.annotation)
                    return item.name, annotations
    return None, {}


def _first_class_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return next(
        (node.name for node in tree.body if isinstance(node, ast.ClassDef)),
        None,
    )


def _extract_examples(description: str) -> list[dict[str, Any]]:
    examples = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", description, re.S | re.I):
        text = html.unescape(re.sub(r"<[^>]+>", "", block))
        match = re.search(
            r"Input:\s*(.*?)\s*Output:\s*(.*?)(?:\s*Explanation:|$)",
            text,
            re.S | re.I,
        )
        if not match:
            continue
        values = _parse_assignments(" ".join(match.group(1).split()))
        if values:
            examples.append(values)
    return examples


def _extract_design_examples(description: str) -> list[dict[str, Any]]:
    examples = []
    decoder = json.JSONDecoder()
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", description, re.S | re.I):
        text = html.unescape(re.sub(r"<[^>]+>", "", block))
        match = re.search(
            r"Input\s*:?[ \t]*\n?\s*(.*?)\s*Output\s*:?[ \t]*\n?\s*(.*?)(?:\s*Explanation|$)",
            text,
            re.S | re.I,
        )
        if not match:
            continue
        try:
            operations, offset = decoder.raw_decode(match.group(1).lstrip())
            arguments, _ = decoder.raw_decode(match.group(1).lstrip()[offset:].lstrip())
            expected, _ = decoder.raw_decode(match.group(2).lstrip())
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            isinstance(operations, list)
            and isinstance(arguments, list)
            and isinstance(expected, list)
            and len(operations) == len(arguments) == len(expected)
            and operations
        ):
            examples.append(
                {"operations": operations, "arguments": arguments, "expected": expected}
            )
    return examples


def _with_design_variants(
    examples: list[dict[str, Any]], class_name: str, has_reference: bool
) -> list[dict[str, Any]]:
    tests = [deepcopy(example) for example in examples]
    first = examples[0]

    # Prefixes exercise state transitions independently of the full published case.
    for end in range(2, len(first["operations"]) + 1):
        if first["expected"][end - 1] is None:
            continue
        tests.append({key: deepcopy(value[:end]) for key, value in first.items()})

    if has_reference:
        # The oracle computes outputs for altered inputs, making these private cases
        # independent of the example's published output.
        for index, args in enumerate(first["arguments"][1:], start=1):
            if len(args) != 1 or not isinstance(args[0], int):
                continue
            for replacement in (0, -1, 1):
                variant = deepcopy(first)
                variant["arguments"][index] = [replacement]
                variant["expected"] = None
                tests.append(variant)
                if len(tests) >= len(examples) + 8:
                    return tests
    elif class_name == "SummaryRanges":
        # addNum is idempotent: replaying it creates a derived case whose expected
        # getIntervals results remain valid even when no Python oracle is supplied.
        variant = deepcopy(first)
        for index in range(len(variant["operations"]) - 1, 0, -1):
            if variant["operations"][index] == "addNum":
                for key in ("operations", "arguments", "expected"):
                    variant[key].insert(index + 1, deepcopy(variant[key][index]))
                break
        tests.append(variant)
    return tests


def _parse_assignments(value: str) -> dict[str, Any] | None:
    normalized = re.sub(r"\bnull\b", "None", value, flags=re.I)
    normalized = re.sub(r"\btrue\b", "True", normalized, flags=re.I)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.I)
    try:
        expression = ast.parse(f"dict({normalized})", mode="eval")
        call = expression.body
        if not isinstance(call, ast.Call) or call.args:
            return None
        parsed = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in call.keywords
            if keyword.arg is not None
        }
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _with_generated_variants(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests = [deepcopy(example) for example in examples]
    first = examples[0]
    for name, value in first.items():
        replacements: list[Any] = []
        if isinstance(value, bool):
            replacements = [not value]
        elif isinstance(value, int):
            replacements = [0, 1]
        elif isinstance(value, str):
            replacements = ["", value[:1], value[::-1]]
        elif isinstance(value, list):
            replacements = [[], value[:1], list(reversed(value))]
            if value and all(isinstance(item, (int, float)) for item in value):
                replacements.append(sorted(value))
        for replacement in replacements:
            variant = deepcopy(first)
            variant[name] = replacement
            if variant not in tests:
                tests.append(variant)
            if len(tests) >= len(examples) + 6:
                return tests
    return tests


def _design_program(
    candidate_code: str,
    reference_code: str | None,
    class_name: str,
    tests: list[dict[str, Any]],
) -> str:
    return f'''from __future__ import annotations
from typing import *
from collections import *
from random import *
import copy, json, math, random, traceback

MARKER = {RESULT_MARKER!r}
CANDIDATE_CODE = {candidate_code!r}
REFERENCE_CODE = {reference_code!r}
CLASS_NAME = {class_name!r}
TESTS = {tests!r}

def canonical(value):
    if isinstance(value, tuple): value = list(value)
    if isinstance(value, list): return [canonical(item) for item in value]
    if isinstance(value, dict): return {{str(key): canonical(item) for key, item in value.items()}}
    return value

def equal(actual, expected):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7)
    return canonical(actual) == canonical(expected)

def load(code, filename):
    namespace = {{
        "defaultdict": defaultdict,
        "Counter": Counter,
        "deque": deque,
        "randint": randint,
        "choice": choice,
    }}
    exec(compile(code, filename, "exec"), namespace)
    cls = namespace.get(CLASS_NAME)
    if not isinstance(cls, type): raise AttributeError(f"missing class {{CLASS_NAME}}")
    return cls

def oracle_sequence(reference_cls, operations, arguments):
    instance = reference_cls(*copy.deepcopy(arguments[0]))
    outputs = [None]
    for index, (operation, args) in enumerate(zip(operations[1:], arguments[1:]), start=1):
        random.seed(104729 + index)
        outputs.append(canonical(getattr(instance, operation)(*copy.deepcopy(args))))
    return outputs

def random_allowed(reference_instance, operation, args, index):
    allowed = []
    for seed_value in range(24):
        clone = copy.deepcopy(reference_instance)
        random.seed(104729 + index + seed_value)
        value = canonical(getattr(clone, operation)(*copy.deepcopy(args)))
        if value not in allowed: allowed.append(value)
    return allowed

try:
    candidate_cls = load(CANDIDATE_CODE, "candidate.py")
    reference_cls = load(REFERENCE_CODE, "reference.py") if REFERENCE_CODE else None
except BaseException:
    traceback.print_exc(); raise

report = {{"planned": len(TESTS), "executed": 0, "skipped": 0, "failures": []}}
for test_index, test in enumerate(TESTS):
    operations, arguments = test["operations"], test["arguments"]
    published = test.get("expected")
    try:
        expected = oracle_sequence(reference_cls, operations, arguments) if reference_cls else published
        if expected is None: raise ValueError("no oracle output")
        candidate = candidate_cls(*copy.deepcopy(arguments[0]))
        reference = reference_cls(*copy.deepcopy(arguments[0])) if reference_cls else None
    except BaseException:
        report["skipped"] += 1
        continue

    report["executed"] += 1
    for operation_index, (operation, args) in enumerate(
        zip(operations[1:], arguments[1:]), start=1
    ):
        try:
            random.seed(104729 + operation_index)
            actual = canonical(getattr(candidate, operation)(*copy.deepcopy(args)))
            wanted = expected[operation_index]
            if reference is not None and operation.lower().startswith("getrandom"):
                allowed = random_allowed(reference, operation, args, operation_index)
                matches = actual in allowed
            else:
                matches = equal(actual, wanted)
            if not matches:
                report["failures"].append({{
                    "test": test_index,
                    "operation": operation_index,
                    "call": operation,
                    "arguments": args,
                    "expected": wanted,
                    "actual": actual,
                }})
                break
            if reference is not None:
                random.seed(104729 + operation_index)
                getattr(reference, operation)(*copy.deepcopy(args))
        except BaseException as exc:
            report["failures"].append({{
                "test": test_index,
                "operation": operation_index,
                "call": operation,
                "error": f"{{type(exc).__name__}}: {{exc}}",
            }})
            break

print(MARKER + json.dumps(report, default=str, separators=(",", ":")))
'''


def _program(
    *,
    candidate_code: str,
    reference_code: str,
    method_name: str,
    annotations: dict[str, str],
    tests: list[dict[str, Any]],
    unordered: bool,
) -> str:
    # Candidate and oracle execute in separate namespaces inside the already
    # resource-limited child process. The candidate never receives oracle output.
    return f'''from __future__ import annotations
from typing import *
import copy, json, math, traceback

MARKER = {RESULT_MARKER!r}
CANDIDATE_CODE = {candidate_code!r}
REFERENCE_CODE = {reference_code!r}
METHOD_NAME = {method_name!r}
ANNOTATIONS = {annotations!r}
TESTS = {tests!r}
UNORDERED = {unordered!r}

class ListNode:
    def __init__(self, val=0, next=None): self.val, self.next = val, next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def list_node(values):
    dummy = ListNode(); current = dummy
    for value in values:
        current.next = ListNode(value); current = current.next
    return dummy.next

def tree_node(values):
    if not values: return None
    nodes = [None if value is None else TreeNode(value) for value in values]
    children = iter(nodes[1:])
    for node in nodes:
        if node is not None:
            node.left = next(children, None); node.right = next(children, None)
    return nodes[0]

def convert(name, value):
    annotation = ANNOTATIONS.get(name, "")
    if "ListNode" in annotation and isinstance(value, list): return list_node(value)
    if "TreeNode" in annotation and isinstance(value, list): return tree_node(value)
    return copy.deepcopy(value)

def canonical(value):
    if isinstance(value, ListNode):
        result = []
        while value is not None and len(result) < 10000:
            result.append(value.val); value = value.next
        return result
    if isinstance(value, TreeNode):
        result, queue = [], [value]
        while queue and len(result) < 10000:
            node = queue.pop(0)
            if node is None: result.append(None); continue
            result.append(node.val); queue.extend([node.left, node.right])
        while result and result[-1] is None: result.pop()
        return result
    if isinstance(value, tuple): value = list(value)
    if isinstance(value, list):
        result = [canonical(item) for item in value]
        if UNORDERED:
            result.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
        return result
    if isinstance(value, dict): return {{str(k): canonical(v) for k, v in value.items()}}
    return value

def invoke(namespace, values):
    cls = namespace.get("Solution")
    target = cls() if cls is not None else namespace
    function = getattr(target, METHOD_NAME, None) if cls is not None else target.get(METHOD_NAME)
    if not callable(function): raise AttributeError(f"missing callable {{METHOD_NAME}}")
    kwargs = {{name: convert(name, value) for name, value in values.items()}}
    result = function(**kwargs)
    if result is None:
        changed = [canonical(kwargs[name]) for name in values]
        result = changed[0] if len(changed) == 1 else changed
    return canonical(result)

base = {{"ListNode": ListNode, "TreeNode": TreeNode}}
candidate_ns, reference_ns = dict(base), dict(base)
report = {{"planned": len(TESTS), "executed": 0, "skipped": 0, "failures": []}}
try:
    exec(compile(CANDIDATE_CODE, "candidate.py", "exec"), candidate_ns)
    exec(compile(REFERENCE_CODE, "reference.py", "exec"), reference_ns)
except BaseException:
    traceback.print_exc(); raise

for index, values in enumerate(TESTS):
    try:
        expected = invoke(reference_ns, values)
    except BaseException:
        report["skipped"] += 1
        continue
    report["executed"] += 1
    try:
        actual = invoke(candidate_ns, values)
        equal = actual == expected
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            equal = math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7)
        if not equal:
            report["failures"].append({{"test": index, "input": values, "expected": expected, "actual": actual}})
    except BaseException as exc:
        report["failures"].append({{"test": index, "input": values, "error": f"{{type(exc).__name__}}: {{exc}}"}})

print(MARKER + json.dumps(report, default=str, separators=(",", ":")))
'''
