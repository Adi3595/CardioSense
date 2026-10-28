import re

def check_latex(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    in_tabular = False
    in_equation = False
    in_algorithmic = False
    errors = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if '\\begin{tabular' in stripped:
            in_tabular = True
            continue
        if '\\end{tabular' in stripped:
            in_tabular = False
            continue
        if '\\begin{equation' in stripped or '\\begin{align' in stripped:
            in_equation = True
            continue
        if '\\end{equation' in stripped or '\\end{align' in stripped:
            in_equation = False
            continue
        if '\\begin{algorithmic' in stripped:
            in_algorithmic = True
            continue
        if '\\end{algorithmic' in stripped:
            in_algorithmic = False
            continue

        # Ignore comments
        code_part = stripped.split('%')[0] if not stripped.startswith('\\%') else stripped

        # Check for unescaped & outside tabular, align, cases
        if not in_tabular and not in_equation:
            unescaped_amp = re.findall(r'(?<!\\)&', code_part)
            if unescaped_amp:
                errors.append(f"Line {i}: Misplaced alignment tab '&': {line}")

        # Check for unescaped % outside comments
        # Check for unescaped #
        if not in_equation:
            # Check for unescaped #
            unescaped_hash = re.findall(r'(?<!\\)#', code_part)
            if unescaped_hash:
                errors.append(f"Line {i}: Unescaped '#' in text: {line}")

    print(f"Verified syntax check on {filepath}: {len(errors)} critical syntax errors found.")
    for e in errors:
        print("  -", e)
    return len(errors)

if __name__ == '__main__':
    import sys
    check_latex(sys.argv[1])
