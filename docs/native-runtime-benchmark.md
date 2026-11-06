# Native runtime: measurements and status

This report covers the migration from llama.cpp to the in-process mistral.rs runtime (`crates/liquid-inference`).
All numbers are from one machine and one run each unless stated, so treat them as indicative, not as a benchmark suite.

## Environment

- Linux x86_64, 2 vCPU, 14 GB RAM, no NVIDIA GPU, no Metal. The CPU is the only device this build can use.
- Models: `LFM3-1.7B-Q4_K_M.gguf`, `LFM3-Omni-2B-Q4_K_M.gguf`, and projectors `mmproj-LFM3-Omni-2B-Q4_K_M.gguf` and `…-Q8_0.gguf`.
- SDK: mistral.rs git rev `52b3c49` (v0.10.0). The LFM3 loader is not in the v0.9.5 tag or crates.io 0.8.1.
- Release builds. The native server is built with `cargo build --release -p liquid-server` (about 5 min cold, 25 s incremental).

## End-to-end: release server, real models, native against legacy

Run with `scripts/native-e2e.mjs` (scratch data folder, random port, real model files symlinked in).

| Check | Native (mistral.rs) | Legacy (llama.cpp) |
| --- | --- | --- |
| Start both models | 7.8–9.9 s | 6.7 s |
| Text chat, one-line answer | 1.6 s | 0.43 s |
| Tool round trip (shell `echo`, full-auto approval) | 5.5–5.9 s | 1.2 s |
| Transcribe the 9.1 s sample clip | 4.7–4.8 s | 0.6 s |
| Read aloud "The weather is clear today." | 9.8–10.1 s (70 KB WAV) | 0.65 s (80 KB WAV) |
| Voice turn: time to first words heard | 4.2–4.7 s | 1.4 s |
| Voice turn: time to first audio | 23–24 s | 1.4 s |
| Voice turn: time to end of reply | 143–158 s (reply of about 40 s of speech) | 4.8 s |

Notes on the voice row: the legacy turn uses the realtime WebSocket and the native turn uses the sequential pipeline,
and the models wrote different replies, so the end-of-reply figures are not the same workload. The time to first
audio is the fair comparison: sentence-level speech starts once the first sentence is written, but on this CPU the
speech model runs slower than real time (see below), so long replies take a long time to finish.

Native passes 10 of 10 checks: both models run in-process, the omni capabilities are reported truthfully, the
engine tile reports `runtime: native` with no download, a tool call runs through the approval path, and the
voice turn heard, answered, and spoke.

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

Peak memory for the 1.7B text model was not captured separately.

## Build and artifact size

| Artifact | Legacy (`HEAD`) | Native |
| --- | --- | --- |
| `liquid-server` release binary | 14.8 MB | 100.4 MB (77.6 MB stripped) |
| Separate engine download | about 30 MB (llama.cpp release) | none |
| Dynamic libraries | glibc, libgcc, libm | the same three: no new shared-library dependency |

The SDK code is most of the increase. Stripping is left to packaging rather than set in the release profile.

## What the runtime does, and what it refuses

- **Quantizations:** Q4_K_M is the only verified small-model quantization. For Omni, Q4_K_M is verified, with the
  Q4_K_M and Q8_0 projectors. Other quantizations are refused with a message ("… is not verified yet; choose Q4_K_M"),
  and the UI disables them. Custom GGUF files go through the SDK, and its error text is shown.
- **Memory:** the SDK's automatic planner reserves about 7 GB for Omni's worst case, whatever the context, and refuses
  to load when that does not fit. The adapter therefore uses a manual device mapping, and the server checks the real
  file sizes against free memory before each load: it needs three times the size of the files, plus 1 GiB.
  Loading is refused with the numbers rather than swapping or crashing.
- **Devices:** CPU only in this build. `gpuLayers: 0` means CPU; other values mean automatic, which is the CPU here.
  The Vulkan heuristic is gone from the native path.
- **Context:** `modelContext` maps to the SDK's maximum model length. Omni was run at 8192 tokens.
- **Speech controls:** the engine has no voice selector and ignores speaker references. Both are reported as unsupported
  in capabilities, the speak response carries `x-liquid-voice-applied: false`, and the Settings controls are disabled.

## Limits that are measured, not assumed

- **Speech is slower than real time on this CPU.** Speaking 3.5 s of audio took 9.5 s. Sentence-level speech in voice
  calls therefore falls behind for long replies. A GPU or Metal build would change this. It is untested here.
- **Voice calls are turn-based** on the native runtime. Full-duplex realtime is not implemented by the SDK, and the
  app labels this in Settings and in the call view. `LIQUID_AGENT_RUNTIME=legacy` restores the llama.cpp realtime call.
- **ASR is less exact with the Q8_0 projector**: "MrstlRS" against "MistralRS" in the same clip.
- **Vision, ASR and speech** were checked on Omni Q4_K_M with its Q4_K_M and Q8_0 projectors, and nowhere else.

## Not verified

- CUDA, Metal, and Apple Silicon. No such hardware was available.
- The LFM3 tool format with every approval mode. Only the full-auto path was run end to end. The existing smoke test
  checks the approval hold; it runs in CI with the native runtime and was not run here.
- Memory under pressure and model switching while active. Both models were loaded at once and unloaded; repeated
  switching under memory pressure was not run.
- Air-gapped operation. It is not verified whether mistral.rs contacts the network when it loads local GGUF files.
  Check this with networking disabled before relying on offline installs.
- Real fresh-install and upgrade runs on a second machine. The settings migration and the untouched engine folder
  are covered by `tests/server/rust-native.test.mjs`.

## RFC status

| RFC item | Status |
| --- | --- |
| Phase 0: spike, compatibility, metrics | Done for CPU text and Omni, except the items listed under "Not verified" |
| Phase 1: `liquid-inference` crate, test adapter, feature flag | Done. The `mistralrs` feature is on in the server. Legacy stays selectable at runtime |
| Phase 2: chat through the native engine | Done. Chat, stop markers, tool parsing, approvals, and idle restore run unchanged |
| Phase 3: Omni vision, ASR, speech | Done for the verified files. Capability probes run real operations |
| Phase 4: voice | Option B taken: turn-based native voice, labelled in the UI, with the limits above. Full-duplex parity is not claimed |
| Phase 5: lifecycle, settings, migration | Done: `runtime` and `runtimeSchemaVersion` markers, idle sleep, leases, no engine download |
| Phase 6: packaging, removal of llama.cpp | Partly done. The engine is not downloaded and the UI says so. The llama.cpp code is kept for rollback (see below) |
| Phase 7: installer/UI cleanup, deletion | Not done for deletion. The removal is gated on the platform tests above |

## Decisions

1. **llama.cpp stays as an explicit rollback.** The RFC requires acceptance on each target platform before deletion,
   and only Linux CPU was available. Deleting the path now would also remove the only faster realtime voice. Deleting
   it later is one change in `backends.rs`, `installer.rs` and the legacy tests, once acceptance passes.
2. **Native is the default for new installs and for upgrades.** The migration only adds markers; models, chats and
   settings are kept, and the engine folder is not touched.
3. **Voice in native mode is turn-based and labelled.** No realtime parity is claimed.
4. **Unverified quantizations are refused**, not offered as working.

## Test status

- `cargo test --workspace --locked`: passes.
- `cargo clippy -- -D warnings` on `liquid-inference` (with and without `mistralrs`) and `liquid-server`: clean.
- `cargo fmt --all -- --check`: clean.
- `npm run typecheck`, `npm run test:web` (27 tests), `npm run build`: pass.
- `npm run test:server` (Node, 12 files): all pass; 5 real-model tests are skipped unless their environment variables are set.
- Real-model tests (`crates/liquid-inference/tests/mistralrs_smoke.rs`, opt-in): text stream, cancel, unload; Omni
  image, ASR and speech. All pass.
- `blaze check -p liquid-inference`: passes. `blaze check` for the whole workspace fails, because blaze cannot load the
  proc-macro chain of the SDK (`macro_rules_attribute`), even with `native-proc-macros` set. Use `cargo` for the native
  paths until blaze supports it.

## Reproduce

```sh
cargo build --release -p liquid-server -p liquid-guardian
LIQUID_E2E_WAV=/path/to/clip.wav node scripts/native-e2e.mjs native
LIQUID_E2E_WAV=/path/to/clip.wav node scripts/native-e2e.mjs legacy   # needs the llama.cpp engine folder

LIQUID_INFERENCE_GGUF=/path/LFM3-1.7B-Q4_K_M.gguf \
  cargo test --release -p liquid-inference --features mistralrs --test mistralrs_smoke text -- --ignored --nocapture
```
