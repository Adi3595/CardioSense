import re

def check_latex(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    in_tabular = False
    in_math = False
    errors = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if '\\begin{tabular' in stripped:
            in_tabular = True
            continue
        if '\\end{tabular' in stripped:
            in_tabular = False
            continue

        # Ignore comments
        code_part = stripped.split('%')[0] if not stripped.startswith('\\%') else stripped

        # Check for unescaped &
        if not in_tabular:
            # Look for & not preceded by \
            unescaped_amp = re.findall(r'(?<!\\)&', code_part)
            if unescaped_amp:
                errors.append(f"Line {i}: Misplaced alignment tab '&' outside tabular: {line}")

        # Check for unescaped _ outside math mode
        # Split by $ to find text mode
        parts = re.split(r'(?<!\\)\$', code_part)
        for part_idx, part in enumerate(parts):
            # Even part_idx are text mode, odd are math mode
            if part_idx % 2 == 0:
                unescaped_underscore = re.findall(r'(?<!\\)_', part)
                if unescaped_underscore:
                    errors.append(f"Line {i}: Unescaped '_' in text mode: {part.strip()}")
                unescaped_hash = re.findall(r'(?<!\\)#', part)
                if unescaped_hash:
                    errors.append(f"Line {i}: Unescaped '#' in text mode: {part.strip()}")

    print(f"Total errors found: {len(errors)}")
    for e in errors:
        print("  -", e)

if __name__ == '__main__':
    import sys
    check_latex(sys.argv[1])
