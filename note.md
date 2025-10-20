# create virtual python environment (venv)
```
python -m venv venv
```

# to active the venv, we must allow scripts for the current shell
```
Set-ExecutionPolicy Bypass -Scope Process
```

# activate the venv
```
.\venv\Scripts\activate
```

# packages to install
```
pip install numpy Pillow torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

# deactivate the venv
```
deactivate
```