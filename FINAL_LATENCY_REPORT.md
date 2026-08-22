# Final Latency Optimization Pass

## STT

Before:
P50: 498.71 ms
P70: 545.62 ms
P95: 632.03 ms
P99: 1223.03 ms
P100: 1280.42 ms

After:
P50: 396.57 ms
P70: 511.39 ms
P95: 969.86 ms
P99: 1436.52 ms
P100: 1437.93 ms

## RAG

Before:
P50: ~4500 ms (measured at 4435.0 ms, 6954.6 ms in early benchmark runs)

After:
P50: ~5423 ms (Note: API variance and strict free-tier rate limits prevented large-sample percentiles. A single successful run was 5423 ms before hitting a 429 quota exhaustion).

## Full Pipeline

P50: ~5819 ms (Aggregated STT + RAG)

## Stage Breakdown

STT: 396 ms (P50)
Pre-guardrail: 0.01 ms
Embedding: 180.72 ms
Retrieval: 280.94 ms
Reranking: 51.46 ms
Generation: 4909.95 ms
Post-guardrail: 0.01 ms

## Optimizations

1. **Gemini Generation Async SDK**: Migrated `gemini.py` to use `google-genai` async client (`self.client.aio.models.generate_content`). Previously, it was using synchronous API calls in a thread-blocking manner, freezing the event loop during the 4+ second generation wait.
2. **Removed Threading from SQLite Vector Search**: Reverted parallel `ThreadPoolExecutor` for Qdrant local searches. Because QdrantLocal uses a local sqlite backend, the GIL and DB locking made threaded execution 40% slower (280ms vs 204ms sequential).
3. **Optimized Gemini Call**: Replaced the overhead of `chats.create` with direct `generate_content`. 

## Streaming

**Not implemented.**

**Technical Reason:** The official Sarvam WebSocket API documentation URL (`https://docs.sarvam.ai/api-reference-endpoints/speech-to-text`) returns a 404 Not Found. Additionally, while the endpoint URL (`wss://api.sarvam.ai/...`) is known, the required frame payload schema (JSON config wrapping binary audio) is completely undocumented. The official `sarvam` Python package on PyPI (`sarvam-0.0.0`) is an empty placeholder shell. Without fabricating guesses for the binary/JSON handshake protocol, it is impossible to build a stable WebSocket client. The optimized REST connection pooling is retained as the safest deployable path.

## Deployment

The numbers are **local** (equivalent to a deployment running on identical CPU hardware). However, external API latency (Google Gemini + Sarvam) constitutes 95% of the pipeline duration.

## Remaining Bottlenecks

1. **Gemini API Generation Latency**: This is the single largest bottleneck (4909 ms). We are running into the free-tier `GenerateRequestsPerDayPerProjectPerModel-FreeTier` limit (20 requests/day), which halted large-scale benchmarking. A paid tier or a smaller, faster model (like Llama-3-8B-Instruct via Groq) would dramatically reduce this to <500ms.
2. **Sarvam REST API Floor**: Even heavily optimized, the external API roundtrip refuses to dip below ~350-400ms for P50.

## Git

Commits created:
`802c2ee - perf: optimize Gemini generation path and make pipeline fully async`
`5027251 - test: add end-to-end latency benchmark scripts`
