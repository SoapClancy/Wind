from pathlib import Path

cwd = Path().cwd()
project_path_ = Path('/'.join(cwd.parts[:cwd.parts.index('MyProject')+1]))
