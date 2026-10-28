# Spike: Qwen 2.5 on WebGPU (issue #1)

A throwaway test page. It is not extension code.

## Run

```sh
npx serve spike
```

Open the URL it prints (for example `http://localhost:3000`) in Chrome.
WebGPU needs a secure context. `localhost` counts as secure.

## Use

1. Pick a model. Click **Load model**. The first load downloads the model.
2. Type a line. Pick a target language. Click **Translate**.
3. Read the numbers in the log box (and in the DevTools console):
   - model load time (first load or cached load)
   - time to first token (TTFT)
   - tokens per second

The page uses `dtype: "q4f16"`. If that fails, it falls back to `"q4"` and logs it.

There is no 3B option: `onnx-community` has no plain `Qwen2.5-3B-Instruct` ONNX build (only Coder and VL).
