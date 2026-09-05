#!/usr/bin/env python3
"""Prove that two versions of a Python file differ only in comments/docstrings.

    python3 tests/check_code_identical.py OLD.py NEW.py

Parses both, strips every docstring, and compares the resulting ASTs dumped
without line numbers. Comments never reach the AST at all, so an empty result
means: whatever changed, it was not code.

Used to make the comment-trimming half of a refactor mechanically checkable
rather than a matter of reading the diff carefully.
"""

import ast
import sys


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            # A function whose whole body is a docstring still needs a
            # statement, so leave a placeholder rather than an empty body.
            node.body = body[1:] or [ast.Pass()]
    return tree


def dump(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    return ast.dump(strip_docstrings(tree), annotate_fields=True,
                    include_attributes=False)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    old, new = dump(argv[1]), dump(argv[2])
    if old == new:
        print("IDENTICAL: %s and %s differ only in comments/docstrings"
              % (argv[1], argv[2]))
        return 0
    print("DIFFERENT: the code itself changed, not just comments", file=sys.stderr)
    # Narrow it down to the first divergence so the caller has somewhere to look.
    for i, (a, b) in enumerate(zip(old, new)):
        if a != b:
            lo = max(0, i - 120)
            print("\nfirst divergence at char %d:" % i, file=sys.stderr)
            print("  old: ...%s" % old[lo:i + 120], file=sys.stderr)
            print("  new: ...%s" % new[lo:i + 120], file=sys.stderr)
            break
    else:
        print("\none is a prefix of the other (length %d vs %d)"
              % (len(old), len(new)), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
