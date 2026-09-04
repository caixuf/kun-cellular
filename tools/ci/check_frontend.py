#!/usr/bin/env python3
"""
tools/ci/check_frontend.py
Frontend Syntax and Asset Integrity Checker
Verifies:
1. All local static assets referenced in frontend/*.html (src, href) exist on disk.
2. All embedded <script> and <script type="module"> blocks parse cleanly with `node --check`.
"""

import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"


def check_html_assets(html_path: Path) -> list[str]:
    errors = []
    content = html_path.read_text(encoding="utf-8")

    # Match src="..." or href="..."
    asset_patterns = [
        r'''(?:src|href)=["']([^"']+)["']''',
    ]

    for pat in asset_patterns:
        for match in re.finditer(pat, content):
            ref = match.group(1).strip()
            # Skip empty, anchor links, data URLs, http(s) URLs, mailto, etc.
            if not ref or ref.startswith(("#", "data:", "http://", "https://", "mailto:", "javascript:")):
                continue

            # Strip URL parameters and hash
            clean_ref = ref.split("?")[0].split("#")[0]
            if not clean_ref:
                continue

            target_path = (html_path.parent / clean_ref).resolve()
            if not target_path.exists():
                errors.append(f"{html_path.name}: Static asset not found: '{ref}' -> '{target_path}'")

    return errors


def check_script_syntax(html_path: Path) -> list[str]:
    errors = []
    content = html_path.read_text(encoding="utf-8")

    # Match script blocks
    script_pattern = re.compile(r'<script\b([^>]*)>(.*?)</script>', re.DOTALL | re.IGNORECASE)

    for idx, match in enumerate(script_pattern.finditer(content)):
        attrs = match.group(1)
        code = match.group(2).strip()

        # If it has src=..., already checked in check_html_assets
        if 'src=' in attrs:
            continue
        # Skip importmap, json, etc.
        if 'type="importmap"' in attrs or 'type="application/json"' in attrs:
            continue
        if not code:
            continue

        is_module = 'type="module"' in attrs
        suffix = ".mjs" if is_module else ".js"

        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", encoding="utf-8", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(code)

        try:
            res = subprocess.run(
                ["node", "--check", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if res.returncode != 0:
                err_lines = res.stderr.strip().split("\n")
                first_few = " | ".join(err_lines[:2])
                errors.append(f"{html_path.name}: Script block #{idx+1} syntax error: {first_few}")
        except Exception as e:
            errors.append(f"{html_path.name}: Failed running node --check: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return errors


def check_es_modules(dir_path: Path) -> list[str]:
    if not dir_path.exists():
        return []
    errors = []
    js_files = sorted(dir_path.glob("*.js"))
    exports_map = {}

    for js_file in js_files:
        content = js_file.read_text(encoding="utf-8")
        exps = set()
        for m in re.finditer(r'export\s+(?:function|const|let|var|class|async\s+function)\s+([a-zA-Z0-9_$]+)', content):
            exps.add(m.group(1))
        for m in re.finditer(r'export\s*\{([^}]+)\}', content):
            for part in m.group(1).split(','):
                part = part.strip()
                if not part:
                    continue
                tokens = re.split(r'\s+as\s+', part)
                exps.add(tokens[-1].strip())
        exports_map[js_file.name] = exps

    for js_file in js_files:
        content = js_file.read_text(encoding="utf-8")
        for m in re.finditer(r'import\s*\{([^}]+)\}\s*from\s*[\'"](\.[^\'"]+)[\'"]', content):
            raw_items = m.group(1)
            raw_target = m.group(2)
            target_name = Path(raw_target).name
            if target_name not in exports_map:
                errors.append(f"{js_file.name}: imports from unknown local module '{raw_target}'")
                continue
            target_exports = exports_map[target_name]
            for item in raw_items.split(','):
                item = item.strip()
                if not item:
                    continue
                orig_symbol = re.split(r'\s+as\s+', item)[0].strip()
                if orig_symbol not in target_exports:
                    errors.append(
                        f"{js_file.name}: imports '{orig_symbol}' from {target_name}, but {target_name} does not export it!"
                    )

    return errors


def main() -> int:
    print("=" * 60)
    print("  KunCellular Frontend Asset & Syntax Integrity Check")
    print(f"  Directory: {FRONTEND_DIR}")
    print("=" * 60)

    all_errors = []
    html_files = sorted(FRONTEND_DIR.glob("*.html"))
    if not html_files:
        print("ERROR: No HTML files found in frontend/ directory!")
        return 1

    for html_path in html_files:
        asset_errs = check_html_assets(html_path)
        all_errors.extend(asset_errs)

        syntax_errs = check_script_syntax(html_path)
        all_errors.extend(syntax_errs)

        status = "OK" if not (asset_errs or syntax_errs) else "FAIL"
        print(f"  [{status}] {html_path.name} (Assets: {len(asset_errs)} errs, Syntax: {len(syntax_errs)} errs)")

    # ES Module Cross-Link Verification for frontend/cellular/
    cellular_dir = FRONTEND_DIR / "cellular"
    esm_errs = check_es_modules(cellular_dir)
    all_errors.extend(esm_errs)
    status = "OK" if not esm_errs else "FAIL"
    print(f"  [{status}] cellular/*.js ES Module Graph ({len(list(cellular_dir.glob('*.js')))} modules, {len(esm_errs)} errs)")

    print("=" * 60)
    if all_errors:
        print(f"FAILED: {len(all_errors)} issues detected:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"SUCCESS: All {len(html_files)} HTML files & ES Module graphs passed integrity check! (Exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
