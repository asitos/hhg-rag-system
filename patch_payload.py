import re

with open("app.py", "r") as f:
    content = f.read()

content = content.replace(
    '"strategy": "semantic"', 
    '"strategy": "semantic", "passage_id": p["id"]'
)

with open("app.py", "w") as f:
    f.write(content)
