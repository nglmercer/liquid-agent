// End-to-end check of the release server with real GGUF models, in a scratch data folder on a random port.
// It never touches a running copy of the app.
//
// Usage: node scripts/native-e2e.mjs [scratch-dir]
// Environment:
//   LIQUID_E2E_MODELS  folder with LFM3-1.7B-Q4_K_M.gguf, LFM3-Omni-2B-Q4_K_M.gguf and
//                      mmproj-LFM3-Omni-2B-Q4_K_M.gguf (default: ~/.local/share/liquid-agent/models)
//   LIQUID_E2E_WAV     a 16-bit WAV clip to transcribe and use as the voice turn
// Build the server first: cargo build --release -p liquid-server
import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import WebSocket from "ws";

const repo = fileURLToPath(new URL("..", import.meta.url));
const [scratchArg] = process.argv.slice(2);
const scratch = scratchArg ?? fs.mkdtempSync(path.join(os.tmpdir(), "liquid-e2e-"));
const models = process.env.LIQUID_E2E_MODELS ?? path.join(os.homedir(), ".local/share/liquid-agent/models");
const wavSource = process.env.LIQUID_E2E_WAV;
if (!wavSource) throw new Error("set LIQUID_E2E_WAV to a 16-bit WAV clip");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const report = [];
const step = (name, ok, detail = "") => {
  report.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? " :: " + detail : ""}`);
};

async function freePort() {
  const s = net.createServer();
  await new Promise((r) => s.listen(0, "127.0.0.1", r));
  const p = s.address().port;
  await new Promise((r) => s.close(r));
  return p;
}

const data = path.join(scratch, "data");
const dist = path.join(scratch, "dist");
fs.rmSync(data, { recursive: true, force: true });
fs.mkdirSync(path.join(data, "models"), { recursive: true });
fs.mkdirSync(dist, { recursive: true });
fs.writeFileSync(path.join(dist, "index.html"), "<html><head></head><body>e2e</body></html>");
for (const f of ["LFM3-1.7B-Q4_K_M.gguf", "LFM3-Omni-2B-Q4_K_M.gguf", "mmproj-LFM3-Omni-2B-Q4_K_M.gguf"]) {
  fs.symlinkSync(path.join(models, f), path.join(data, "models", f));
}
fs.writeFileSync(
  path.join(data, "settings.json"),
  JSON.stringify({
    smallQuant: "Q4_K_M", omniQuant: "Q4_K_M", omniProjector: "Q4_K_M",
    startupMode: "dual", language: "en", uiLanguage: "en",
    temperature: 0.3, approvalMode: "fullAuto", modelContext: 8192,
  }),
);

const port = await freePort();
const base = `http://127.0.0.1:${port}`;
const env = {
  ...process.env,
  LIQUID_AGENT_NO_AUTOSTART: "1",
  LIQUID_AGENT_RUST_HOME: data,
  LIQUID_AGENT_WEB_DIST: dist,
  LIQUID_AGENT_RUST_PORT: String(port),
};
const server = spawn(path.join(repo, "target/release/liquid-server"), [], { env, stdio: ["ignore", "pipe", "pipe"] });
let serverLog = "";
server.stdout.on("data", (d) => (serverLog += d));
server.stderr.on("data", (d) => (serverLog += d));
const logPath = path.join(scratch, "server.log");

let token = "";
for (let i = 0; i < 100 && !token; i++) {
  try {
    const html = await (await fetch(base)).text();
    token = html.match(/name="liquid-token" content="([^"]+)"/)?.[1] || "";
  } catch {}
  if (!token) await sleep(100);
}
if (!token) throw new Error("server did not start: " + serverLog);

async function api(route, method = "GET", body, raw) {
  const headers = { "x-liquid-token": token };
  let payload;
  if (raw !== undefined) {
    headers["content-type"] = "audio/wav";
    payload = raw;
  } else if (body !== undefined) {
    headers["content-type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(base + route, { method, headers, body: payload });
  return res;
}
async function json(route, method, body) {
  const res = await api(route, method, body);
  if (!res.ok) throw new Error(`${route}: ${res.status} ${await res.text()}`);
  return res.json();
}
async function waitFor(read, done, timeoutMs = 240000) {
  const end = Date.now() + timeoutMs;
  let last;
  while (Date.now() < end) {
    last = await read();
    if (done(last)) return last;
    await sleep(200);
  }
  const brief = last?.parts ? last.parts.filter((p) => ["small", "omni"].includes(p.id)).map((p) => ({ id: p.id, status: p.status, error: p.error, text: p.text })) : last;
  throw new Error("timed out; last=" + JSON.stringify(brief).slice(0, 900));
}
const part = (state, id) => state.parts.find((p) => p.id === id);

try {
  // 1. Parts start in-process.
  let t = Date.now();
  await json("/api/parts/small/start", "POST");
  await json("/api/parts/omni/start", "POST");
  const both = await waitFor(
    () => json("/api/state"),
    (s) => part(s, "small")?.status === "running" && part(s, "omni")?.status === "running",
  );
  step("start both models", true, `${Date.now() - t} ms`);
  const caps = part(both, "omni").capabilities;
  step("omni capabilities reported", caps?.vision === true && caps?.asr === true && caps?.tts === true && caps?.voiceSelection === false, JSON.stringify(caps));
  step("no engine download or engine part", !both.parts.some((p) => p.id === "engine"), "");

  // 2. Text chat, thinking off: the reply must be an answer, not reasoning.
  t = Date.now();
  const chat = await json("/api/chats", "POST", { text: "In one short sentence, what is the capital of France?" });
  const done = await waitFor(
    () => json("/api/chats").then((list) => list.find((c) => c.id === chat.id)),
    (c) => ["done", "error", "stopped"].includes(c?.messages?.at(-1)?.status),
  );
  const reply = done.messages.at(-1);
  step("text chat completes", reply.status === "done" && /paris/i.test(reply.content), `${Date.now() - t} ms; status=${reply.status}; content=${JSON.stringify(reply.content).slice(0, 200)}`);

  // 3. Tool-call round trip: ask for a tool that needs no approval, then check the steps.
  t = Date.now();
  const toolChat = await json("/api/chats", "POST", { text: "Use the shell tool to run `echo liquid-e2e-ok` and tell me the output." });
  const toolDone = await waitFor(
    () => json("/api/chats").then((list) => list.find((c) => c.id === toolChat.id)),
    (c) => ["done", "error", "stopped"].includes(c?.messages?.at(-1)?.status),
  );
  const toolReply = toolDone.messages.at(-1);
  const steps = toolReply.steps || [];
  step("tool chat completes", toolReply.status === "done", `${Date.now() - t} ms; steps=${JSON.stringify(steps.map((s) => ({ label: s.label, status: s.status }))).slice(0, 300)}; content=${JSON.stringify(toolReply.content).slice(0, 200)}`);

  // 4. ASR: transcribe the sample clip.
  const wav = fs.readFileSync(wavSource);
  t = Date.now();
  const asr = await api("/api/transcribe", "POST", undefined, wav);
  const asrJson = asr.ok ? await asr.json() : { error: await asr.text() };
  step("transcribe endpoint", asr.ok && /mystral|mistral/i.test(asrJson.text || ""), `${Date.now() - t} ms; ${JSON.stringify(asrJson).slice(0, 220)}`);

  // 5. Speech: read aloud, voice selection reported as not applied.
  t = Date.now();
  const speak = await api("/api/speak", "POST", { text: "The weather is clear today.", voice: "male" });
  const speechBytes = speak.ok ? Buffer.from(await speak.arrayBuffer()) : Buffer.alloc(0);
  const applied = speak.headers.get("x-liquid-voice-applied");
  step("speak endpoint returns WAV", speak.ok && speechBytes.toString("ascii", 0, 4) === "RIFF", `${Date.now() - t} ms; bytes=${speechBytes.length}; voice-applied=${applied}; status=${speak.status}`);
  fs.writeFileSync(path.join(scratch, "speak.wav"), speechBytes);

  // 6. Capability probe: real operations, honest flags.
  const probe = await json("/api/capabilities/omni", "POST");
  step("capability probe", probe.runtime === "native", JSON.stringify(probe));

  // 7. Voice turn: send the clip as PCM16 at 16 kHz, then commit.
  const pcm16k = await pcmFromWav16k(wav);
  t = Date.now();
  const voice = new WebSocket(`ws://127.0.0.1:${port}/voice?t=${token}`, { origin: base });
  await new Promise((r, j) => { voice.once("open", r); voice.once("error", j); });
  const events = [];
  const binaries = [];
  const marks = {};
  voice.on("message", (data, isBinary) => {
    const now = Date.now();
    if (isBinary) {
      binaries.push(Buffer.from(data));
      marks.firstAudio ??= now;
    } else {
      const e = JSON.parse(data.toString());
      events.push(e);
      if (e.type === "user") marks.heard ??= now;
      if (e.type === "assistant") marks.firstText ??= now;
      if (e.type === "tool") marks.tool ??= now;
      if (e.type === "turn.done") marks.done ??= now;
    }
  });
  voice.send(JSON.stringify({ type: "start" }));
  await waitFor(async () => events.find((e) => e.type === "ready"), (v) => !!v, 20000);
  for (let off = 0; off < pcm16k.length; off += 16000) voice.send(pcm16k.subarray(off, Math.min(off + 16000, pcm16k.length)));
  const committedAt = Date.now();
  voice.send(JSON.stringify({ type: "commit" }));
  await waitFor(async () => events.some((e) => e.type === "turn.done"), (v) => v, 240000);
  const ready = events.find((e) => e.type === "ready");
  const user = events.find((e) => e.type === "user");
  const assistant = events.filter((e) => e.type === "assistant").at(-1);
  const audioFormat = events.find((e) => e.type === "audio.format");
  const ms = (k) => (marks[k] ? `${marks[k] - committedAt} ms` : "n/a");
  step("voice turn heard, answered, spoke", !!user && !!assistant?.text && binaries.length > 0 && !!audioFormat, `turnBased=${ready?.turnBased}; realtimeFullDuplex=${ready?.realtimeFullDuplex}; heard@${ms("heard")} firstText@${ms("firstText")} firstAudio@${ms("firstAudio")} done@${ms("done")}; heard=${JSON.stringify(user?.text)?.slice(0, 80)}; said=${JSON.stringify(assistant?.text)?.slice(0, 120)}; audioBytes=${binaries.reduce((a, b) => a + b.length, 0)}; rate=${audioFormat?.rate}`);
  voice.close();

  // 8. Stop both: unloaded, states back to installed.
  await json("/api/parts/small/stop", "POST");
  await json("/api/parts/omni/stop", "POST");
  const stopped = await waitFor(() => json("/api/state"), (s) => part(s, "small")?.status === "installed" && part(s, "omni")?.status === "installed");
  step("stop both models", !!stopped, "");
} catch (err) {
  step("e2e run", false, String(err.stack || err));
} finally {
  fs.writeFileSync(logPath, serverLog);
  server.kill();
  await sleep(300);
  const failed = report.filter((r) => !r.ok).length;
  console.log(`SUMMARY: ${report.length - failed}/${report.length} passed`);
}

// Reads a 16-bit mono WAV of any rate and returns 16 kHz PCM16 by linear resampling.
async function pcmFromWav16k(buf) {
  let pos = 12, rate = 16000, channels = 1, bits = 16, data = null;
  while (pos + 8 <= buf.length) {
    const id = buf.toString("ascii", pos, pos + 4);
    const size = buf.readUInt32LE(pos + 4);
    if (id === "fmt ") { channels = buf.readUInt16LE(pos + 10); rate = buf.readUInt32LE(pos + 12); bits = buf.readUInt16LE(pos + 22); }
    if (id === "data") { data = buf.subarray(pos + 8, pos + 8 + size); break; }
    pos += 8 + size + (size % 2);
  }
  if (!data || bits !== 16) throw new Error("unsupported wav");
  const frames = data.length / (2 * channels);
  const out = Buffer.alloc(Math.floor((frames * 16000) / rate) * 2);
  for (let i = 0; i < out.length / 2; i++) {
    const src = (i * rate) / 16000;
    const a = Math.floor(src), b = Math.min(a + 1, frames - 1), f = src - a;
    const sa = data.readInt16LE(a * 2 * channels), sb = data.readInt16LE(b * 2 * channels);
    out.writeInt16LE(Math.round(sa + (sb - sa) * f), i * 2);
  }
  return out;
}
