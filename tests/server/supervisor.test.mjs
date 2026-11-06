import { describe, it, expect } from "vitest";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// The release supervisor (`liquid-agent supervise`) in a scratch install: the real debug server and launcher,
// its own port and home, the fixture engine. Needs: cargo build -p liquid-server -p liquid-guardian.
const repo = fileURLToPath(new URL("../..", import.meta.url));
const guardian = path.join(repo, "target", "debug", "liquid-guardian");
const serverBin = path.join(repo, "target", "debug", "liquid-server");
const ready = process.platform === "linux" && fs.existsSync(guardian) && fs.existsSync(serverBin);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

describe.skipIf(!ready)("Release supervisor", () => {
  it("restarts a crashed server, leaves a clean stop alone, follows an update, and stops with the launcher", async () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), "liquid-supervisor-"));
    const root = path.join(home, "install");
    const data = path.join(home, "data");
    const dist = path.join(home, "dist");
    for (const dir of [path.join(root, "app", "bin"), path.join(root, "bin"), data, dist]) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dist, "index.html"), "<html><head></head><body>supervisor</body></html>");
    const arch = process.arch === "arm64" ? "arm64" : "x64";
    const server = path.join(root, "app", "bin", `liquid-server-linux-${arch}`);
    fs.symlinkSync(serverBin, server);
    // The launcher must be called liquid-agent; a copy keeps that name for current_exe().
    const launcher = path.join(root, "bin", "liquid-agent");
    fs.copyFileSync(guardian, launcher);
    fs.chmodSync(launcher, 0o755);

    const socket = net.createServer();
    await new Promise((resolve) => socket.listen(0, "127.0.0.1", resolve));
    const port = socket.address().port;
    await new Promise((resolve) => socket.close(resolve));
    fs.writeFileSync(path.join(root, "bin", "launcher.json"), JSON.stringify({ root, data, port: String(port), repo: "none", branch: "none" }));

    const env = {
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: path.join(home, "xdg"),
      LIQUID_AGENT_NO_OPEN: "1",
      LIQUID_AGENT_NO_AUTOSTART: "1",
      LIQUID_AGENT_NATIVE_ENGINE: "fixture",
      LIQUID_AGENT_WEB_DIST: dist,
      LIQUID_AGENT_SUPERVISOR_GRACE_SECS: "4",
    };
    delete env.LIQUID_AGENT_SUPERVISE;
    delete env.DISPLAY;
    delete env.WAYLAND_DISPLAY;
    const run = (args) => {
      const result = spawnSync(launcher, args, { env, encoding: "utf8", timeout: 30_000 });
      if (result.status !== 0) throw new Error(`liquid-agent ${args.join(" ")} failed: ${result.stderr || result.stdout}`);
      return result.stdout;
    };
    const healthy = async () => {
      try {
        return (await (await fetch(`http://127.0.0.1:${port}/healthz`)).json()).app === "liquid-agent";
      } catch {
        return false;
      }
    };
    const waitFor = async (what, check, ms = 30_000) => {
      const end = Date.now() + ms;
      while (Date.now() < end) {
        if (await check()) return;
        await sleep(200);
      }
      throw new Error(`timed out waiting for ${what}`);
    };
    const pidOf = (file) => Number(fs.readFileSync(path.join(data, file), "utf8").trim());
    const log = () => fs.readFileSync(path.join(data, "logs", "server.log"), "utf8");
    const recorded = new Set();
    const serverPid = () => {
      const pid = pidOf("server.pid");
      recorded.add(pid);
      return pid;
    };

    try {
      // 1. open starts the supervisor, which starts the server.
      run(["open"]);
      await waitFor("the server", healthy);
      const supervisor = pidOf("supervisor.pid");
      recorded.add(supervisor);
      expect(alive(supervisor)).toBe(true);
      const first = serverPid();

      // 2. A crash (SIGKILL) is restarted.
      process.kill(first, "SIGKILL");
      await waitFor("the restarted server", async () => (await healthy()) && serverPid() !== first);
      expect(log()).toMatch(/supervisor: the server stopped unexpectedly \(signal 9\); restarting it in 1 s/);

      // 3. An update handover: a clean stop, then a new server the supervisor did not start. It follows that one,
      //    so a later crash of the new server is restarted too.
      const second = serverPid();
      process.kill(second, "SIGTERM");
      await waitFor("the clean stop", async () => !alive(second));
      expect(fs.existsSync(path.join(data, "server.clean-exit"))).toBe(true);
      const handover = spawn(server, [], {
        env: { ...env, LIQUID_AGENT_PORT: String(port), LIQUID_AGENT_HOME: data, LIQUID_AGENT_ROOT: path.join(root, "app") },
        stdio: "ignore",
        detached: true,
      });
      recorded.add(handover.pid);
      handover.unref();
      await waitFor("the handed-over server", async () => (await healthy()) && serverPid() === handover.pid);
      expect(fs.existsSync(path.join(data, "server.clean-exit"))).toBe(false);
      await sleep(1500); // let the supervisor notice and follow it
      process.kill(handover.pid, "SIGKILL");
      await waitFor("the server after the handed-over one crashed", async () => (await healthy()) && serverPid() !== handover.pid);
      expect(alive(supervisor)).toBe(true);

      // 4. A clean stop with no update: nothing restarts it, and the supervisor leaves after the grace period.
      const third = serverPid();
      process.kill(third, "SIGTERM");
      await waitFor("the supervisor to leave", async () => !alive(supervisor), 15_000);
      expect(await healthy()).toBe(false);
      expect(log()).toMatch(/no server came back; the supervisor stops/);

      // 5. stop ends both, and nothing comes back.
      run(["open"]);
      await waitFor("the server again", healthy);
      const again = pidOf("supervisor.pid");
      recorded.add(again);
      serverPid();
      run(["stop"]);
      await waitFor("the stop", async () => !(await healthy()) && !alive(again));
      await sleep(3000);
      expect(await healthy()).toBe(false);

      // 6. Opted out: open starts the server directly, with no supervisor.
      spawnSync(launcher, ["open"], { env: { ...env, LIQUID_AGENT_SUPERVISE: "0" }, timeout: 30_000 });
      await waitFor("the unsupervised server", healthy);
      serverPid();
      expect(fs.existsSync(path.join(data, "supervisor.pid"))).toBe(false);
      run(["stop"]);
      await waitFor("the unsupervised stop", async () => !(await healthy()));
    } finally {
      spawnSync(launcher, ["stop"], { env, stdio: "ignore", timeout: 10_000 });
      // Only processes of this scratch install: a recorded PID may since belong to something else.
      const ours = (pid) => {
        try {
          return fs.readFileSync(`/proc/${pid}/cmdline`, "utf8").includes(home);
        } catch {
          return false;
        }
      };
      for (const pid of recorded) if (pid > 1 && alive(pid) && ours(pid)) process.kill(pid, "SIGKILL");
      fs.rmSync(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
    }
  }, 180_000);
});
