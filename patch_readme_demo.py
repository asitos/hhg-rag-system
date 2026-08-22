import re

with open("README.md", "r") as f:
    content = f.read()

new_demo_section = """## Demo

Run the video-ready interactive demo (runs the real local RAG pipeline with mock deterministic APIs for offline recording):

```bash
./scripts/run_demo.sh
```

Then open your browser to `http://localhost:8000`.

Recommended demo flow:
1. Select **English RAG (Default)** in the Scenario dropdown and ask: *"What is a corporation?"*
2. Select **English RAG (Default)** in the Scenario dropdown and ask: *"How do mutual funds work?"*
3. Select **Hindi RAG** in the Scenario dropdown and ask: *"भारत की राजधानी क्या है?" (What is the capital of India?)*
4. Show transcript, retrieval, reranking, sources, and latency for the queries above.
5. Select **Off-Topic Guardrail** in the dropdown and ask: *"What is the weather today?"* to demonstrate safety blocks.
"""

content = re.sub(r'## Demo.*?5\. Show grounded answer.*?(?=\n\n##|\Z)', new_demo_section, content, flags=re.DOTALL)

with open("README.md", "w") as f:
    f.write(content)
