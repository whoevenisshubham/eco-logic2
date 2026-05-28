import py_compile, glob, sys

files = glob.glob('**/*.py', recursive=True)
errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print('OK', f)
    except Exception as e:
        print('ERR', f, e)
        errors.append((f, str(e)))

print('\nSummary:')
print('Total files:', len(files))
print('Errors:', len(errors))
if errors:
    for f, e in errors:
        print('-', f, ':', e)
sys.exit(1 if errors else 0)
