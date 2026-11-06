# Native runtime: measurements and status

Liquid Agent runs its models in-process through the mistral.rs SDK (`crates/liquid-inference`). The llama.cpp engine,
its download, its localhost model servers and the legacy realtime voice path have been removed.

The measurements below were taken on one machine, one run each unless stated. They are indicative, not a benchmark
suite. The llama.cpp column is from the same machine before the removal, kept as the before/after evidence.

## Environment

- Linux x86_64, AMD BC-250 (6 cores, 12 threads; the shell's `OMP_NUM_THREADS=2` made `nproc` report 2, but candle
  ignores it and uses the 6 physical cores), 14 GB RAM, AMD GPU (no CUDA, Metal or Vulkan support in the SDK). The CPU
  is the only device available here.
- Models: `LFM3-1.7B-Q4_K_M.gguf`, `LFM3-Omni-2B-Q4_K_M.gguf`, and projectors `mmproj-LFM3-Omni-2B-Q4_K_M.gguf` and `…-Q8_0.gguf`.
- SDK: mistral.rs git rev `52b3c49` (v0.10.0). The LFM3 loader is not in the v0.9.5 tag or crates.io 0.8.1.

## End-to-end: release server, real models

Run with `scripts/native-e2e.mjs` (scratch data folder, random port, real model files linked in).

| Check | Native (mistral.rs) | llama.cpp (before removal) |
| --- | --- | --- |
| Start both models | 7.8–10 s | 6.7 s |
| Text chat, one-line answer | 1.6 s | 0.43 s |
| Tool round trip (shell `echo`, full-auto approval) | 5.5–5.9 s | 1.2 s |
| Transcribe the 9.1 s sample clip | 4.5–4.8 s | 0.6 s |
| Read aloud "The weather is clear today." | 9.7–10.1 s (70 KB WAV) | 0.65 s (80 KB WAV) |
| Voice turn: time to first words heard | 4.1–4.7 s | 1.4 s |
| Voice turn: time to first audio | 22–24 s | 1.4 s |
| Voice turn: time to end of reply | 140–158 s (reply of about 40 s of speech) | 4.8 s |

The final run on the code without llama.cpp passes 10 of 10 checks: both models run in-process, the omni capabilities
are reported truthfully, no engine part exists, a tool call runs through the approval path, and the voice turn heard,
answered, and spoke.

The voice rows need care. The llama.cpp turn used its realtime WebSocket and the native turn uses the turn-based
pipeline, and the models wrote different replies, so the end-of-reply figures are not the same workload. The time to
first audio is the fair comparison. Sentences are spoken as soon as each is written, but on this CPU the speech model
runs slower than real time, so long replies take a long time to finish.

## Phase 0 probes (SDK directly, CPU)

| Probe | Result |
| --- | --- |
| LFM3-1.7B Q4_K_M load | 3.5 s |
| LFM3-1.7B Q4_K_M first token (21-token prompt) | 451 ms |
| LFM3-1.7B Q4_K_M decode | 23.0 tok/s (prefill 46.9 tok/s) |
| Omni Q4_K_M + Q4_K_M projector load | 6.4 s |
| Omni Q4_K_M + Q4_K_M projector first token | 502 ms, 22.6 tok/s decode |
| Omni image question (64×64 solid red) | "Red", 1.9 s |
| Omni ASR of the 9.1 s clip | Accurate, 4.7 s |
| Omni speech, streamed, PCM16 at 24 kHz | First audio chunk at 3.1 s; 3.5 s of speech in 9.5 s total |
| Round trip: speech generated, then transcribed | "The weather is clear today." (exact) |
| Cancel: next request's first response after a dropped stream | 263 ms (text), 310 ms (Omni) |
| Compute type on CPU (LFM3-1.7B) | F32 fastest: 351 ms first token, 20.8 tok/s; F16 548 ms, 17.2 tok/s; BF16 421 ms, 16.1 tok/s |
| Peak resident memory, Omni Q4 + projector | 5.8 GB |
| Omni Q4_K_M with the Q8_0 projector | Image, ASR and speech all work; ASR is less exact ("MrstlRS") |

## CPU tuning (1.7B Q4_K_M, probe, decode tok/s)

| Threads (`RAYON_NUM_THREADS`) | Decode | First token |
| --- | --- | --- |
| 4 | 17.5 | 597 ms |
| 6 (candle's default: physical cores) | 14.7–20.7 | 492–964 ms |
| 12 (all SMT threads) | 6.7–8.8 | 673–696 ms |

The default is the fastest; using SMT threads more than halves decode speed. Compute type `Auto` already picks F32
on the CPU, the fastest measured, and candle selects its AVX2/FMA kernels at run time, so no `target-cpu` flag is
needed.

## Debug builds

The tensor crates (candle, gemm, mistralrs-quant, half) are built at `opt-level = 3` in the dev profile. Server, text
model only, same prompts:

| Build | Load | Chat 1 | Chat 2 |
| --- | --- | --- | --- |
| Debug, before (unoptimised kernels) | 30.9 s | 128.9 s | 109.4 s |
| Debug, now | 6.4 s | 2.5 s | 2.1 s |
| Release | 3.5 s | 1.8 s | 1.4 s |

## Build and artifact size

| Artifact | llama.cpp (before) | Native (now) |
| --- | --- | --- |
| `liquid-server` release binary | 14.8 MB | 99.5 MB (about 78 MB stripped) |
| Separate engine download | about 30 MB (llama.cpp release) | none |
| Shared libraries | glibc, libgcc, libm | glibc, libstdc++, libgcc, libm |

The SDK is most of the increase. The new shared library, `libstdc++`, comes from `esaxx-rs`, a C++ component of the
tokenizer library the SDK uses. Packaging has to ship it or require it. Removing that dependency needs a change
upstream, and the release notes must list it.

Measured: a release build from clean takes about 5 minutes, and an incremental release build about 25 seconds.

## What the runtime does, and what it refuses

- **Quantizations:** Q4_K_M is the only verified text-model quantization. For Omni, Q4_K_M is verified with the
  Q4_K_M and Q8_0 projectors. Other quantizations are refused with a message ("… is not verified yet; choose
  Q4_K_M"), and the UI disables them. Custom GGUF files go through the SDK, and its error text is shown.
- **Memory:** the SDK's automatic planner reserves about 7 GB for Omni's worst case, whatever the context, and refuses
  to load when that does not fit. The adapter therefore uses a manual device mapping, and the server checks the real
  file sizes against free memory before each load: it needs three times the size of the files, plus 1 GiB. A refused
  load says so with the numbers rather than swapping or crashing.
- **Devices:** chosen at load. `Auto` (the default) tries CUDA, then Metal, then the CPU. A backend is only tried if
  the build has it (`--features cuda` / `metal` on `liquid-server`) and the device opens; a GPU load that fails falls
  back to the next device, and the reasons are logged. `LIQUID_AGENT_DEVICE=cpu|cuda|metal` forces one, and an
  explicit GPU that is not available is refused with the reason instead of quietly running on the CPU. Diagnostics
  report each model's device and `inferenceDevices`, the devices `Auto` can use. A CUDA build links `libcuda` at
  start-up, so it is a separate artifact for machines with the NVIDIA driver. Vulkan is not possible in process: neither
  mistral.rs nor candle has a Vulkan backend.
- **Context:** `modelContext` maps to the SDK's maximum model length. Omni was run at 8192 tokens.
- **Startup modes:** hybrid and automatic modes keep the text model loaded alongside Omni when the memory estimate
  allows it, as the README describes. Eviction happens only when headroom is insufficient.
- **Speech controls:** the engine has no voice selector and ignores speaker references. Both are reported as
  unsupported in capabilities, the speak response carries `x-liquid-voice-applied: false`, and the voice control is
  disabled in Settings.
- **Upgrades:** the settings migration (schema 2) removes the llama.cpp keys and the old `engine` folder, and keeps
  models, chats and every other setting. The engine folder is removed only when it holds the old server program or
  its version marker.

## Limits that are measured, not assumed

- **Speech is slower than real time on this CPU.** Speaking 3.5 s of audio took 9.5 s. Sentence-level speech in voice
  calls falls behind for long replies. A GPU or Metal build would change this. It is untested here.
- **Voice calls are turn-based.** Full-duplex realtime is not implemented by the SDK, and the app labels this in
  Settings and during a call.
- **Vision, ASR and speech** were checked on Omni Q4_K_M with its Q4_K_M and Q8_0 projectors, and nowhere else.

## Not verified

- CUDA and Metal builds. The device selection and CPU fallback are tested in the CPU build (an explicit GPU is
  refused, `Auto` ends at the CPU), but no NVIDIA or Apple hardware or CUDA toolkit was available, so a `cuda` or
  `metal` build has not been compiled or run, and GPU speed is not measured.
- Memory under pressure and model switching while a model is active. Both models were loaded at once and unloaded;
  repeated switching under memory pressure was not run.
- Air-gapped operation. It is not verified whether mistral.rs contacts the network when it loads local GGUF files.
  Check this with networking disabled before relying on offline installs.
- Fresh installs and upgrades on a second machine. The settings migration and the engine-folder removal are covered
  by `tests/server/rust-native.test.mjs`, on this machine only.

## Status against the RFC

| RFC item | Status |
| --- | --- |
| Phase 0: spike, compatibility, metrics | Done for CPU text and Omni, except the items listed under "Not verified" |
| Phase 1: `liquid-inference` crate, adapter, scripted engine for tests | Done |
| Phase 2: chat through the native engine | Done. Chat, stop markers, tool parsing, approvals, idle sleep and restore run unchanged |
| Phase 3: Omni vision, ASR, speech | Done for the verified files. Capability probes run real operations |
| Phase 4: voice | Turn-based native voice, labelled in the UI, with the limits above. Full-duplex parity is not claimed |
| Phase 5: lifecycle, settings, migration | Done: schema marker, idle sleep, leases, quantization checks, no engine download |
| Phase 6: packaging and removal | Done in code. The llama.cpp engine, its download, the localhost model servers and the legacy realtime path are removed. Packaging must account for `libstdc++` |
| Phase 7: installer, UI and cleanup | Done. The only remaining "llama" words are the GGUF architecture name in the metadata parser’s test data, and the README sentence that says there is no llama.cpp engine |
| Acceptance on Metal and CUDA, rollback between releases | Not verified on this machine |

## Decisions

1. **llama.cpp is removed.** The RFC makes deletion conditional on platform acceptance, and only Linux CPU was
   available here. The user asked for the full replacement. The platform acceptance is therefore an open item, and the
   removed code is in git history if a rollback is ever needed.
2. **Native is the only runtime.** Upgrades migrate the settings in place and keep the models.
3. **Voice is turn-based and labelled.** No realtime parity is claimed.
4. **Unverified quantizations are refused**, not offered as working.
5. **The model's wire format is kept** where it is useful: the tool parser, prompts and approvals are the same code.

## Test status

- `cargo test --workspace --locked`: 96 tests pass.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean. `cargo fmt --all -- --check`: clean.
- `npm run typecheck`, `npm run test:web` (27 tests), `npm run build`: pass.
- `npm run test:server` (Node, 12 files): all pass. 18 tests run and 5 real-model tests are skipped unless their
  environment variables are set.
- Opt-in real-model tests (`crates/liquid-inference/tests/mistralrs_smoke.rs`, and `scripts/native-e2e.mjs` for the
  whole server): pass on this machine.
- `blaze check -p liquid-inference`: passes. `blaze check` for the whole workspace fails, because blaze cannot load the
  SDK's proc-macro chain (`macro_rules_attribute`). Use `cargo` for the server until blaze supports it.

## Reproduce

```sh
cargo build --release -p liquid-server -p liquid-guardian
LIQUID_E2E_WAV=/path/to/clip.wav node scripts/native-e2e.mjs

LIQUID_INFERENCE_GGUF=/path/LFM3-1.7B-Q4_K_M.gguf \
  cargo test --release -p liquid-inference --features mistralrs --test mistralrs_smoke text -- --ignored --nocapture
```
