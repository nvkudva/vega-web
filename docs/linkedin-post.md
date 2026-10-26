Vega 0.8B now runs fully in the browser. No API, no server, nothing leaves the tab.

Vega isn't a chat model. It's a typed decision model by Nandakishor M at Convai Innovations: give it a state and typed questions (choice, score, yes/no), get back calibrated probabilities with an abstain flag.

I ported it to onnxruntime-web: frozen Qwen3.5 in 8-bit plus its physics engine, tokenize to answer in TypeScript. Text and image states both work.

Try it live in your tab:

Links
Live demo: https://vega-onnx.pages.dev/
Weights: https://huggingface.co/nvkudva/vega-onnx-q8

#OnDeviceAI #OpenWeights #AIEngineering
