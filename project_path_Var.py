import os
import re

project_path_ = os.path.dirname(os.path.abspath(__file__))
project_path_ = os.path.abspath(os.path.join(project_path_, "../"))
project_path_ = project_path_ + '/'
project_path_ = re.sub(r'\\', '/', project_path_)
