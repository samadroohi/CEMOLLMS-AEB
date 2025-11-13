import pathlib
root = pathlib.Path(r"d:\Samad\Python\CEMOLLMS-AEB")
for path in root.rglob('*.py'):
    data = path.read_bytes()
    if b'\x00' in data:
        print(path)
        break
else:
    print('NO_NULLS')
