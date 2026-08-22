import re

with open("gradio_app.py", "r") as f:
    content = f.read()

content = content.replace('"strategy": "demo"', '"strategy": "semantic"')

with open("gradio_app.py", "w") as f:
    f.write(content)

