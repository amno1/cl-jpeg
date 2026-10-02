#!/usr/bin/env python3
"""
tools/audit-symbol-shadowing.py

Automated mechanical audit for case-insensitive symbol shadowing in Common Lisp source files.
Common Lisp's standard reader folds symbols to uppercase (or is case-insensitive), meaning
that binding 'H' (horizontal sampling) in an outer or inner scope and 'h' (height) in a nested
scope results in silent variable shadowing or mutation.

This script parses Common Lisp forms, tracks lexical variable scopes (parameters, let, let*,
loop for/with), and flags any shadowed identifiers across scopes. Exits with 1 if violations
are found, 0 if clean.
"""

import sys
import os

def parse_lisp_tokens(text):
    tokens = []
    i = 0
    n = len(text)
    line = 1
    while i < n:
        c = text[i]
        if c == '\n':
            line += 1
            i += 1
        elif c in ' \t\r':
            i += 1
        elif c == ';':
            while i < n and text[i] != '\n':
                i += 1
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == '\\':
                    i += 2
                else:
                    if text[i] == '\n':
                        line += 1
                    i += 1
            i += 1
        elif c in '()':
            tokens.append((c, line))
            i += 1
        else:
            start = i
            while i < n and text[i] not in '() \t\r\n;"':
                i += 1
            tokens.append((text[start:i], line))
    return tokens

def tokens_to_tree(tokens):
    stack = [[]]
    for tok, line in tokens:
        if tok == '(':
            new_list = []
            stack[-1].append((new_list, line))
            stack.append(new_list)
        elif tok == ')':
            if len(stack) > 1:
                stack.pop()
        else:
            stack[-1].append((tok, line))
    return stack[0]

def extract_symbol_name(item):
    """Extract symbol string and line from an item, handling (str, line) or ([items], line)."""
    if isinstance(item, tuple):
        if isinstance(item[0], str):
            return item[0], item[1]
        elif isinstance(item[0], list) and len(item[0]) > 0:
            return extract_symbol_name(item[0][0])
    return None, None

def audit_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    tokens = parse_lisp_tokens(text)
    tree = tokens_to_tree(tokens)
    violations = []

    def analyze_func(form):
        if not isinstance(form, tuple) or not isinstance(form[0], list):
            return
        lst, line = form
        if not lst:
            return
        first = lst[0]
        if isinstance(first, tuple) and isinstance(first[0], str) and first[0].lower() in ['defun', 'defmacro']:
            func_name = lst[1][0] if len(lst) > 1 and isinstance(lst[1], tuple) and isinstance(lst[1][0], str) else 'unknown'
            params_form = lst[2] if len(lst) > 2 and isinstance(lst[2], tuple) and isinstance(lst[2][0], list) else None
            
            scope_stack = [{}] # List of dicts: var_name.lower() -> (original_name, line_no)
            if params_form:
                for p in params_form[0]:
                    p_name, pline = extract_symbol_name(p)
                    if p_name and not p_name.startswith('&') and not p_name.startswith(':'):
                        scope_stack[0][p_name.lower()] = (p_name, pline)
            
            def walk(node):
                if not isinstance(node, tuple) or not isinstance(node[0], list):
                    return
                nl, nline = node
                if not nl:
                    return
                head = nl[0]
                pushed = False
                if isinstance(head, tuple) and isinstance(head[0], str):
                    hname = head[0].lower()
                    if hname in ['let', 'let*']:
                        bindings = nl[1] if len(nl) > 1 and isinstance(nl[1], tuple) and isinstance(nl[1][0], list) else None
                        if bindings:
                            new_scope = {}
                            for b in bindings[0]:
                                bname, bline = extract_symbol_name(b)
                                if bname:
                                    b_low = bname.lower()
                                    # Check for collision in all active parent scopes
                                    for s in reversed(scope_stack):
                                        if b_low in s:
                                            orig_name, orig_line = s[b_low]
                                            violations.append((filepath, func_name, line, bname, bline, orig_name, orig_line))
                                            break
                                    new_scope[b_low] = (bname, bline)
                            scope_stack.append(new_scope)
                            pushed = True
                    elif hname == 'loop':
                        new_scope = {}
                        idx = 1
                        while idx < len(nl):
                            item = nl[idx]
                            if isinstance(item, tuple) and isinstance(item[0], str):
                                w = item[0].lower()
                                if w in ['for', 'with']:
                                    if idx + 1 < len(nl):
                                        vname, vline = extract_symbol_name(nl[idx+1])
                                        if vname:
                                            v_low = vname.lower()
                                            for s in reversed(scope_stack):
                                                if v_low in s:
                                                    orig_name, orig_line = s[v_low]
                                                    violations.append((filepath, func_name, line, vname, vline, orig_name, orig_line))
                                                    break
                                            new_scope[v_low] = (vname, vline)
                            idx += 1
                        scope_stack.append(new_scope)
                        pushed = True
                
                for child in nl:
                    walk(child)
                if pushed:
                    scope_stack.pop()

            for bform in lst[3:]:
                walk(bform)

    for top in tree:
        analyze_func(top)

    return violations

def main():
    files = sys.argv[1:]
    if not files:
        files = ['jpeg.lisp', 'simd.lisp']
    
    total_violations = 0
    print("=== Checking for Case-Insensitive Symbol Shadowing ===")
    for path in files:
        if not os.path.exists(path):
            continue
        vlist = audit_file(path)
        if vlist:
            print(f"\n[FAIL] Found {len(vlist)} symbol shadowing violation(s) in {path}:")
            for fpath, func, fline, var, vline, orig_var, orig_line in vlist:
                print(f"  - {fpath}:{vline} in {func} (L{fline}): symbol '{var}' shadows '{orig_var}' bound at L{orig_line}")
            total_violations += len(vlist)
        else:
            print(f"[PASS] {path}: No symbol shadowing detected.")

    if total_violations > 0:
        print(f"\nERROR: Total {total_violations} shadowing violation(s) detected!")
        sys.exit(1)
    else:
        print("\nAll audited files are clean. No variable shadowing found.")
        sys.exit(0)

if __name__ == '__main__':
    main()
