import { describe, it } from "vitest";
import { spawn } from "node:child_process";
import { once } from "node:events";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Autostart on the in-process model: installed models start with the app, according to the startup mode, and a
// model whose files are missing is not started.

describe("Rust autostart starts installed models", () => {
  it("Rust autostart starts installed models", async () => {
    const repo = fileURLToPath(new URL("../..", import.meta.url));
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "liquid-autostart-"));
    const data = path.join(dir, "data");
    const dist = path.join(dir, "dist");
    fs.mkdirSync(dist);
    fs.writeFileSync(path.join(dist, "index.html"), "<head></head>");
    fs.mkdirSync(path.join(data, "models"), { recursive: true });
    fs.writeFileSync(path.join(data, "models", "LFM3-1.7B-Q4_K_M.gguf"), "fixture");
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    async function free() {
      const s = net.createServer();
      await new Promise((resolve) => s.listen(0, "127.0.0.1", resolve));
      const port = s.address().port;
      await new Promise((resolve) => s.close(resolve));
      return port;
    }
    const port = await free();
    const base = `http://127.0.0.1:${port}`;
    const env = {
      ...process.env,
      LIQUID_AGENT_RUST_HOME: data,
      LIQUID_AGENT_WEB_DIST: dist,
      LIQUID_AGENT_RUST_PORT: String(port),
      LIQUID_AGENT_NATIVE_ENGINE: "fixture",
    };
    delete env.LIQUID_AGENT_NO_AUTOSTART;
    const check = (ok, message) => {
      if (!ok) throw Error(message);
      console.log(`✓ ${message}`);
    };
    // Starts the server and returns its parts once autostart has settled.
    async function startAndRead(expectRunning) {
      const app = spawn(path.join(repo, "target/debug/liquid-server"), [], { env, stdio: ["ignore", "pipe", "pipe"] });
      let stderr = "";
      app.stderr.on("data", (d) => (stderr += d));
      let token = "";
      for (let i = 0; i < 100; i++) {
        try {
          const html = await (await fetch(base)).text();
          token = html.match(/name="liquid-token" content="([^"]+)"/)?.[1] || "";
          if (token) break;
        } catch {}
        await sleep(100);
      }
      if (!token) throw Error(stderr);
      let parts = [];
      for (let i = 0; i < 300; i++) {
        const state = await (await fetch(`${base}/api/state`, { headers: { "x-liquid-token": token } })).json();
        parts = state.parts;
        if (expectRunning.every((id) => parts.find((p) => p.id === id)?.status === "running")) break;
        await sleep(100);
      }
      return { app, parts };
    }
    const stop = async (app) => {
      const exited = once(app, "exit");
      app.kill("SIGINT");
      await exited;
    };
    try {
      let { app, parts } = await startAndRead(["small"]);
      check(parts.find((p) => p.id === "small").status === "running", "Autostart starts the installed text model");
      await stop(app);

      // Both models installed, dual mode: both start with the app.
      fs.writeFileSync(path.join(data, "models", "LFM3-Omni-2B-Q4_K_M.gguf"), "fixture");
      fs.writeFileSync(path.join(data, "models", "mmproj-LFM3-Omni-2B-Q8_0.gguf"), "fixture");
      fs.writeFileSync(path.join(data, "settings.json"), JSON.stringify({ startupMode: "dual" }));
      ({ app, parts } = await startAndRead(["small", "omni"]));
      check(parts.find((p) => p.id === "small").status === "running" && parts.find((p) => p.id === "omni").status === "running", "Dual mode starts both installed models");
      await stop(app);

      // Omni without its projector is not installed, so only the text model starts.
      fs.unlinkSync(path.join(data, "models", "mmproj-LFM3-Omni-2B-Q8_0.gguf"));
      fs.writeFileSync(path.join(data, "settings.json"), JSON.stringify({ startupMode: "auto" }));
      ({ app, parts } = await startAndRead(["small"]));
      check(parts.find((p) => p.id === "small").status === "running" && parts.find((p) => p.id === "omni").status !== "running", "Automatic startup does not start a model whose files are missing");
      await stop(app);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }, 600_000);
});
