# TranslateMe

A browser extension (Chrome, Edge, Brave) that translates YouTube captions into your language.
The translation runs **100% locally** in your browser with a small AI model on WebGPU.

- Free. No API keys. No server.
- Private. Your data never leaves your computer.
- The model downloads one time (~1 GB) and stays cached.

## How it works

```
YouTube captions → join into sentences → local model (WebGPU, offscreen document) → subtitle overlay
```

1. You turn on CC. The extension reads the caption track that the YouTube player loads.
2. It joins caption chunks into full sentences.
3. A local model (Qwen 2.5 via Transformers.js) translates ahead of the playhead.
4. An overlay shows the translated text in sync with the video.

## Status

Early development. See the [issues](../../issues) and [milestones](../../milestones) for the plan.

## Stack

- Manifest V3
- Vite + TypeScript + CRXJS
- Transformers.js (ONNX Runtime Web, WebGPU)

## V1 languages

Bulgarian, Spanish, German, English.

## License

MIT
