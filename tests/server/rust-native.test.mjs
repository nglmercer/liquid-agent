import { describe, it, expect } from "vitest";
import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Native runtime, on the scripted fixture engine (LIQUID_AGENT_NATIVE_ENGINE=fixture). No model weights and no
// llama.cpp binary are involved: the test proves that chat, state and lifecycle work with no engine download.

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function freePort() {
  const s = net.createServer();
  await new Promise((resolve) => s.listen(0, "127.0.0.1", resolve));
  const port = s.address().port;
  await new Promise((resolve) => s.close(resolve));
  return port;
}

describe("Native runtime (fixture engine)", () => {
  it("serves chat and part lifecycle without any engine download", async () => {
    const repo = fileURLToPath(new URL("../..", import.meta.url));
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "liquid-native-"));
    const dist = path.join(dir, "dist");
    const data = path.join(dir, "data");
    fs.mkdirSync(dist);
    fs.mkdirSync(path.join(data, "models"), { recursive: true });
    fs.writeFileSync(path.join(dist, "index.html"), "<html><head></head><body>test</body></html>");
    // Only the presence of these files matters to the fixture engine.
    fs.writeFileSync(path.join(data, "models", "LFM3-1.7B-Q4_K_M.gguf"), "fixture weights");
    fs.writeFileSync(path.join(data, "models", "LFM3-Omni-2B-Q4_K_M.gguf"), "fixture weights");
    fs.writeFileSync(path.join(data, "models", "mmproj-LFM3-Omni-2B-Q4_K_M.gguf"), "fixture projector");
    // An install from before the native runtime: no runtime key, and a llama.cpp folder that must survive.
    fs.mkdirSync(path.join(data, "engine"));
    fs.writeFileSync(path.join(data, "engine", ".version"), "b12017");
    fs.writeFileSync(
      path.join(data, "settings.json"),
      JSON.stringify({ smallQuant: "Q4_K_M", omniQuant: "Q4_K_M", omniProjector: "Q4_K_M", startupMode: "dual" }),
    );
    const port = await freePort();
    const base = `http://127.0.0.1:${port}`;
    const env = {
      ...process.env,
      LIQUID_AGENT_RUNTIME: "native",
      LIQUID_AGENT_NATIVE_ENGINE: "fixture",
      LIQUID_AGENT_NO_AUTOSTART: "1",
      LIQUID_AGENT_RUST_HOME: data,
      LIQUID_AGENT_WEB_DIST: dist,
      LIQUID_AGENT_RUST_PORT: String(port),
    };
    const app = spawn(path.join(repo, "target/debug/liquid-server"), [], { env, stdio: ["ignore", "pipe", "pipe"] });
    let log = "";
    app.stderr.on("data", (d) => (log += d));
    app.stdout.on("data", (d) => (log += d));
    let token = "";
    try {
      for (let i = 0; i < 100 && !token; i++) {
        try {
          const html = await (await fetch(base)).text();
          token = html.match(/name="liquid-token" content="([^"]+)"/)?.[1] || "";
        } catch {}
        if (!token) await sleep(100);
      }
      if (!token) throw new Error(log);

      const api = async (route, method = "GET", body) => {
        const res = await fetch(base + route, {
          method,
          headers: { "x-liquid-token": token, "content-type": "application/json" },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
        if (!res.ok) throw new Error(`${route}: ${res.status} ${await res.text()}`);
        return res.json();
      };
      const waitFor = async (read, done, what) => {
        for (let i = 0; i < 300; i++) {
          const value = await read();
          if (done(value)) return value;
          await sleep(50);
        }
        throw new Error(`timed out waiting for ${what}`);
      };

      // Upgrade: the migration marks the settings and keeps the old engine folder, untouched.
      const migrated = await api("/api/settings");
      expect(migrated.runtime).toBe("native");
      expect(migrated.runtimeSchemaVersion).toBe(1);
      expect(migrated.smallQuant).toBe("Q4_K_M");
      expect(fs.readFileSync(path.join(data, "engine", ".version"), "utf8")).toBe("b12017");

      // The engine tile is built in: nothing is downloaded or required.
      const state = await api("/api/state");
      const engine = state.parts.find((p) => p.id === "engine");
      expect(engine.status).toBe("installed");
      expect(engine.runtime).toBe("native");

      // Start the text model in-process; it reports its capabilities truthfully.
      await api("/api/parts/small/start", "POST");
      const running = await waitFor(
        () => api("/api/state").then((s) => s.parts.find((p) => p.id === "small")),
        (part) => part.status === "running",
        "the text model to start",
      );
      expect(running.runtime).toBe("native");
      expect(running.capabilities.text).toBe(true);
      expect(running.capabilities.voiceSelection).toBe(false);
      expect(running.capabilities.speakerReference).toBe(false);

      // A chat turn streams the fixture answer through the native path.
      const created = await api("/api/chats", "POST", { text: "Say hello" });
      expect(created.id).toBeTruthy();
      const finished = await waitFor(
        () => api("/api/chats").then((list) => list.find((c) => c.id === created.id)),
        (chat) => chat?.messages?.at(-1)?.status === "done",
        "the chat reply",
      );
      expect(finished.messages.at(-1).content).toBe("Hello world");

      // Diagnostics say native, and report unknown GPU memory as null rather than zero.
      const diagnostics = await api("/api/diagnostics");
      expect(diagnostics.runtime).toBe("native");
      expect(diagnostics.engineProperties.small.gpuMemoryBytes).toBeNull();

      // Stop unloads the model and returns the part to installed.
      await api("/api/parts/small/stop", "POST");
      const stopped = await waitFor(
        () => api("/api/state").then((s) => s.parts.find((p) => p.id === "small")),
        (part) => part.status === "installed",
        "the text model to stop",
      );
      expect(stopped.status).toBe("installed");
    } finally {
      app.kill();
      await sleep(100);
    }
  }, 120_000);
});
