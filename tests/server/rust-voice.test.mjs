import { describe, it } from "vitest";
import { spawn } from "node:child_process";
import { once } from "node:events";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import WebSocket from "ws";

// Voice calls on the in-process model. A call is turn-based: the browser sends the caller's audio, the server
// transcribes it, answers through the normal chat, and streams the reply back as PCM16 at 24 kHz. The model's
// replies are scripted, so the test checks the protocol and the saved chat, not the wording.

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function freePort() {
  const s = net.createServer();
  await new Promise((resolve) => s.listen(0, "127.0.0.1", resolve));
  const port = s.address().port;
  await new Promise((resolve) => s.close(resolve));
  return port;
}

describe("Rust voice calls on the in-process model", () => {
  it("Rust voice call hears a turn, answers it, speaks it, and saves it", async () => {
    const repo = fileURLToPath(new URL("../..", import.meta.url));
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "liquid-voice-"));
    const data = path.join(dir, "data");
    const dist = path.join(dir, "dist");
    fs.mkdirSync(dist);
    fs.writeFileSync(path.join(dist, "index.html"), "<html><head></head><body>voice</body></html>");
    fs.mkdirSync(path.join(data, "models"), { recursive: true });
    for (const file of ["LFM3-1.7B-Q4_K_M.gguf", "LFM3-Omni-2B-Q4_K_M.gguf", "mmproj-LFM3-Omni-2B-Q8_0.gguf"]) {
      fs.writeFileSync(path.join(data, "models", file), "fixture");
    }
    const rules = [{ when: "", reply: "Hello voice.", delayMs: 0 }];
    const scriptPath = path.join(dir, "native-script.json");
    fs.writeFileSync(scriptPath, JSON.stringify(rules));
    const requestLog = path.join(dir, "model-request.jsonl");
    const port = await freePort();
    const base = `http://127.0.0.1:${port}`;
    const env = {
      ...process.env,
      LIQUID_AGENT_NO_AUTOSTART: "1",
      LIQUID_AGENT_RUST_HOME: data,
      LIQUID_AGENT_WEB_DIST: dist,
      LIQUID_AGENT_RUST_PORT: String(port),
      LIQUID_AGENT_NATIVE_ENGINE: "fixture",
      LIQUID_AGENT_NATIVE_SCRIPT: scriptPath,
      LIQUID_AGENT_NATIVE_REQUEST_LOG: requestLog,
    };
    const app = spawn(path.join(repo, "target/debug/liquid-server"), [], { env, stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    app.stderr.on("data", (chunk) => (stderr += chunk));
    const check = (ok, message) => {
      if (!ok) throw new Error(message);
      console.log(`✓ ${message}`);
    };
    let token = "";
    let voice;
    try {
      for (let i = 0; i < 100 && !token; i++) {
        try {
          const html = await (await fetch(base)).text();
          token = html.match(/name="liquid-token" content="([^"]+)"/)?.[1] || "";
        } catch {}
        if (!token) await sleep(100);
      }
      if (!token) throw new Error(stderr);
      const api = async (route, method = "GET", body) => {
        const res = await fetch(base + route, {
          method,
          headers: { "x-liquid-token": token, "content-type": "application/json" },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
        if (!res.ok) throw new Error(`${route}: ${res.status} ${await res.text()}`);
        return res.json();
      };
      await api("/api/settings", "PATCH", { realtimeLanguage: "es", language: "en", startupMode: "dual" });

      const rejected = await new Promise((resolve, reject) => {
        const bad = new WebSocket(`ws://127.0.0.1:${port}/voice?t=${token}`, { origin: "https://evil.example" });
        bad.on("unexpected-response", (_request, response) => {
          resolve(response.statusCode);
          response.resume();
        });
        bad.on("open", () => reject(new Error("foreign origin accepted")));
        bad.on("error", reject);
      });
      check(rejected === 401, "Voice WebSocket rejects foreign origins");

      await api("/api/parts/omni/start", "POST");
      for (let i = 0; i < 200; i++) {
        if ((await api("/api/state")).parts.find((p) => p.id === "omni").status === "running") break;
        await sleep(100);
      }

      voice = new WebSocket(`ws://127.0.0.1:${port}/voice?t=${token}`, { origin: base });
      await once(voice, "open");
      const events = [];
      const audio = [];
      voice.on("message", (data, isBinary) => {
        if (isBinary) audio.push(Buffer.from(data));
        else events.push(JSON.parse(data.toString()));
      });
      const waitFor = async (predicate, what) => {
        for (let i = 0; i < 300; i++) {
          const found = events.find(predicate);
          if (found) return found;
          await sleep(50);
        }
        throw new Error(`voice event timed out: ${what}`);
      };
      voice.send(JSON.stringify({ type: "start" }));
      const ready = await waitFor((e) => e.type === "ready", "ready");
      check(
        ready.inputRate === 16000 && ready.outputRate === 24000 && ready.turnBased === true && ready.realtimeFullDuplex === false && ready.configurationConfirmed === true,
        "Voice call reports 16 kHz input, 24 kHz output and that it is turn-based",
      );
      check(ready.effectiveLanguage === "es", "Voice call uses the saved call language");

      // One second of a quiet tone as the caller's speech, then the end of the turn.
      const speech = Buffer.alloc(16000 * 2);
      for (let i = 0; i < 16000; i++) speech.writeInt16LE(Math.round(2000 * Math.sin((i / 16000) * 440 * 2 * Math.PI)), i * 2);
      for (let off = 0; off < speech.length; off += 16000) voice.send(speech.subarray(off, Math.min(off + 16000, speech.length)));
      voice.send(JSON.stringify({ type: "commit" }));

      const user = await waitFor((e) => e.type === "user", "heard");
      check(user.text === "hello from the test", "Voice call transcribes the caller's turn");
      const assistant = await waitFor((e) => e.type === "assistant" && e.text === "Hello voice.", "reply");
      check(assistant.text === "Hello voice.", "Voice call answers through the chat");
      const format = await waitFor((e) => e.type === "audio.format", "audio format");
      check(format.rate === 24000, "Voice call speaks at 24 kHz");
      await waitFor((e) => e.type === "turn.done", "turn done");
      check(audio.reduce((n, b) => n + b.length, 0) > 0, "Voice call streams the spoken reply as audio");

      const chats = await api("/api/chats");
      const saved = chats.find((c) => c.id === ready.chatId);
      check(saved?.messages.some((m) => m.role === "assistant" && m.content === "Hello voice." && m.status === "done"), "Voice turns are saved to the chat history");
      check(fs.readFileSync(requestLog, "utf8").includes("input_audio"), "The caller's audio reaches the model for transcription");
    } finally {
      voice?.terminate();
      const exited = once(app, "exit");
      app.kill("SIGINT");
      await exited;
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }, 300_000);
});
