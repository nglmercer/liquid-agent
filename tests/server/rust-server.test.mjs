import { describe, it, expect } from "vitest";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { once } from "node:events";
import WebSocket from "ws";

// Migrated from scripts/rust-server-check.mjs. The checks throw on failure; vitest reports each one.

describe("Native server API, security and state", () => {
  it("Native server API, security and state", async () => {
    const repo = fileURLToPath(new URL("../..", import.meta.url));
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "liquid-rust-server-"));
    const dist = path.join(dir,"dist"); const data = path.join(dir,"data"); fs.mkdirSync(dist);
    fs.mkdirSync(data);
    fs.mkdirSync(path.join(data,"uploads"));
    const existingFile = path.join(data,"uploads","file.txt");
    fs.writeFileSync(existingFile,"old attachment");
    const outsideFile = path.join(dir,"outside.txt"); fs.writeFileSync(outsideFile,"private");
    fs.writeFileSync(path.join(data,"uploads.json"), JSON.stringify({file:{id:"file",name:"file.txt",mime:"text/plain",kind:"text",size:14,path:existingFile,at:1}, unsafe:{id:"unsafe",name:"outside.txt",mime:"text/plain",kind:"text",size:7,path:outsideFile,at:1}}));
    fs.writeFileSync(path.join(data,"memory.json"), JSON.stringify(["Already saved fact."]));
    fs.writeFileSync(path.join(data,"routines.v2.json"), JSON.stringify([{id:"existing",name:"Saved",prompt:"Do work",icon:"sun",color:"#ffffff",schedule:{enabled:false,time:"08:00",days:[1]},lastRun:0}]));
    fs.writeFileSync(path.join(data,"chats.json"), JSON.stringify([{id:"oldchat",title:"Old",createdAt:1,updatedAt:1,pinned:false,messages:[{id:"m1",role:"assistant",content:"",status:"streaming",steps:[{status:"running"}]},{id:"m2",role:"user",content:"file",attachments:[{id:"file"}]}]},{id:"sharedchat",title:"Shared",createdAt:1,updatedAt:1,pinned:false,messages:[{role:"user",content:"file",attachments:[{id:"file"}]}]},{id:"unsafechat",title:"Unsafe",createdAt:1,updatedAt:1,pinned:false,messages:[{role:"user",content:"file",attachments:[{id:"unsafe"}]}]}]));
    fs.writeFileSync(path.join(dist,"index.html"), "<html><head></head><body>test</body></html>");
    fs.writeFileSync(path.join(dir,"secret.txt"), "not public");
    fs.symlinkSync(path.join(dir,"secret.txt"), path.join(dist,"escape.txt"));
    const sock = net.createServer(); await new Promise((resolve) => sock.listen(0,"127.0.0.1",resolve));
    const modelSock = net.createServer(); await new Promise((resolve) => modelSock.listen(0,"127.0.0.1",resolve));
    const omniSock = net.createServer(); await new Promise((resolve) => omniSock.listen(0,"127.0.0.1",resolve));
    const browserSock = net.createServer(); await new Promise((resolve) => browserSock.listen(0,"127.0.0.1",resolve));
    const guardianSock = net.createServer(); await new Promise((resolve) => guardianSock.listen(0,"127.0.0.1",resolve));
    const port = sock.address().port; const modelPort = modelSock.address().port; const omniPort = omniSock.address().port; const guardianPort = guardianSock.address().port; const browserPort = browserSock.address().port;
    await new Promise((resolve) => sock.close(resolve)); await new Promise((resolve) => modelSock.close(resolve)); await new Promise((resolve) => omniSock.close(resolve)); await new Promise((resolve) => guardianSock.close(resolve)); await new Promise((resolve) => browserSock.close(resolve));
    fs.mkdirSync(path.join(data,"engine")); fs.mkdirSync(path.join(data,"models"));
    fs.writeFileSync(path.join(data,"models","LFM3-1.7B-Q4_K_M.gguf"),"test weights");
    fs.writeFileSync(path.join(data,"models","LFM3-Omni-2B-Q4_K_M.gguf"),"test weights");
    fs.writeFileSync(path.join(data,"models","mmproj-LFM3-Omni-2B-Q8_0.gguf"),"test projector");
    fs.mkdirSync(path.join(data,"guardian"));
    for (const filename of ["config.json","model.safetensors","tokenizer.json"]) fs.writeFileSync(path.join(data,"guardian",filename),"test guardian");
    fs.writeFileSync(path.join(data,"guardian",".ready"),"ProCreations/auto-150m-3");
    const bridgeDir=path.join(data,"bridge"); fs.mkdirSync(bridgeDir,{recursive:true});
    fs.writeFileSync(path.join(bridgeDir,".ready"),"");
    // The real Web helper browser: Chrome for Testing's headless shell, from LIQUID_TEST_CHROME or the Playwright cache.
    function findChrome() {
      if (process.env.LIQUID_TEST_CHROME) return process.env.LIQUID_TEST_CHROME;
      const cache = path.join(os.homedir(), ".cache", "ms-playwright");
      const found = fs.existsSync(cache) ? fs.readdirSync(cache).filter((d) => d.startsWith("chromium_headless_shell-")).sort().reverse() : [];
      for (const dir of found) {
        const bin = path.join(cache, dir, "chrome-headless-shell-linux64", "chrome-headless-shell");
        if (fs.existsSync(bin)) return bin;
      }
      return null;
    }
    const chrome = findChrome();
    if (chrome) { fs.mkdirSync(path.join(bridgeDir,"browser","shell"),{recursive:true}); fs.symlinkSync(chrome, path.join(bridgeDir,"browser","shell","chrome-headless-shell")); }
    // The guardian is a separate helper process. Its classify endpoint is served by this small script.
    const guardianScript = path.join(dir,"guardian-helper");
    fs.writeFileSync(guardianScript, `#!${process.execPath}
    const http = require('http'); const fs = require('fs');
    const port = Number(process.argv[3]);
    http.createServer((q,s)=>{
      if(q.method !== 'POST') return s.end(JSON.stringify({status:'ok'}));
      let body=''; q.on('data',chunk=>body+=chunk); q.on('end',()=>{
        if(q.url === '/classify') { fs.writeFileSync(${JSON.stringify(path.join(dir,"guardian-request.json"))},body); const mode=fs.readFileSync(${JSON.stringify(path.join(dir,"guardian-mode"))},'utf8'); s.writeHead(200,{'Content-Type':'application/json'}); if(mode==='slow'||mode==='delayed') { const timer=setTimeout(()=>s.end(JSON.stringify({p_deny:0.1})),mode==='slow'?5000:500); s.on('close',()=>clearTimeout(timer)); return; } return s.end(JSON.stringify({p_deny:mode==='low'?0.1:mode==='high'?0.85:2})); }
        s.writeHead(404); s.end();
      });
    }).listen(port,'127.0.0.1');
    `);
    fs.chmodSync(guardianScript,0o755);
    // The text and Omni models run in-process. Their scripted replies and the requests they receive are recorded here.
    const nativeRules = [
      { when: "delayed write file", reply: `<tool_call><function=write_file><parameter=path>${path.join(dir,"agent-output.txt")}</parameter><parameter=content>approved content</parameter></function></tool_call>`, delayMs: 400 },
      { when: "write file", reply: `<tool_call><function=write_file><parameter=path>${path.join(dir,"agent-output.txt")}</parameter><parameter=content>approved content</parameter></function></tool_call>`, delayMs: 0 },
      { when: "dangerous shell", reply: "<tool_call><function=shell><parameter=command>sudo true</parameter></function></tool_call>", delayMs: 0 },
      { when: "fetch local page", reply: "<tool_call><function=fetch_url><parameter=url>http://127.0.0.1:12345/</parameter></function></tool_call>", delayMs: 0 },
      { when: "browser test click", reply: "<tool_call><function=browser_click><parameter=target>Submit</parameter></function></tool_call>", delayMs: 0 },
      { when: "browser test fail", reply: "<tool_call><function=browser_click><parameter=target>Nonexistent thing</parameter></function></tool_call>", delayMs: 0 },
      { when: "browser test read", reply: "<tool_call><function=browser_read></function></tool_call>", delayMs: 0 },
      { when: "browser test type", reply: "<tool_call><function=browser_type><parameter=target>Search box</parameter><parameter=text>hello</parameter></function></tool_call>", delayMs: 0 },
      { when: "browser test local", reply: "<tool_call><function=browser_goto><parameter=url>http://127.0.0.1:12345/</parameter></function></tool_call>", delayMs: 0 },
      { when: "try tool", reply: "Hello <tool_call><function=shell>", delayMs: 0 },
      { when: "please wait", reply: "Hello friend!", delayMs: 5000 },
      { when: "", reply: "Hello friend!", delayMs: 50 },
    ];
    const scriptPath = path.join(dir,"native-script.json");
    fs.writeFileSync(scriptPath, JSON.stringify(nativeRules));
    const requestLog = path.join(dir,"model-request.jsonl");
    fs.writeFileSync(path.join(dir,"guardian-mode"),"low");
    const base = `http://127.0.0.1:${port}`;
    const env = { ...process.env,LIQUID_AGENT_NO_AUTOSTART:"1", LIQUID_AGENT_RUST_PORT:String(port), LIQUID_AGENT_RUST_HOME:data, LIQUID_AGENT_WEB_DIST:dist, LIQUID_AGENT_NATIVE_ENGINE:"fixture", LIQUID_AGENT_NATIVE_SCRIPT:scriptPath, LIQUID_AGENT_NATIVE_REQUEST_LOG:requestLog, LIQUID_GUARDIAN_PORT:String(guardianPort), LIQUID_AGENT_GUARDIAN_BINARY:guardianScript, LIQUID_AGENT_BROWSER_PORT:String(browserPort), TEST_BROWSER_LOG:path.join(dir,"browser.log"), TEST_BROWSER_POLICY:path.join(dir,"browser-policy.json") };
    let child;
    let socket;
    async function start() {
      child = spawn(path.join(repo,"target/debug/liquid-server"),[],{env,stdio:["ignore","pipe","pipe"]});
      let log = ""; child.stderr.on("data",(chunk) => log += chunk);
      for (let i=0;i<100;i++) { try { if ((await fetch(`${base}/healthz`)).ok) return; } catch {} await new Promise((resolve) => setTimeout(resolve,100)); }
      throw new Error(`Rust server failed to start: ${log}`);
    }
    async function stop() { const exited = once(child,"exit"); child.kill("SIGINT"); await exited; child = null; }
    function check(condition,message) { if (!condition) throw new Error(message); console.log(`✓ ${message}`); }
    function lastRequest() { const lines = fs.readFileSync(path.join(dir,"model-request.jsonl"),"utf8").trim().split("\n"); return JSON.parse(lines.at(-1)); }
    try {
      await start();
      const page = await (await fetch(base)).text();
      const token = page.match(/name="liquid-token" content="([^"]+)"/)?.[1];
      check(Boolean(token),"Rust server injects its session token");
      socket = new WebSocket(`ws://127.0.0.1:${port}/ws?t=${token}`, { origin: base });
      const firstMessage = once(socket,"message");
      const [helloBytes] = await firstMessage;
      const hello = JSON.parse(helloBytes.toString());
      check(hello.type === "hello" && hello.memory[0] === "Already saved fact." && hello.routines[0].id === "existing", "Rust WebSocket sends the saved state");
      check(hello.chats[0].messages[0].status === "stopped" && hello.chats[0].messages[0].steps[0].result === "Interrupted.", "Rust server recovers interrupted saved chats");
      const rejected = await new Promise((resolve, reject) => {
        const bad = new WebSocket(`ws://127.0.0.1:${port}/ws?t=${token}`, { origin: "https://evil.example" });
        bad.on("unexpected-response", (_request, response) => { resolve(response.statusCode); response.resume(); });
        bad.on("open", () => { bad.close(); reject(new Error("foreign WebSocket origin accepted")); });
        bad.on("error", reject);
      });
      check(rejected === 401, "Rust WebSocket rejects foreign origins");
      const missingToken = await new Promise((resolve, reject) => {
        const bad = new WebSocket(`ws://127.0.0.1:${port}/ws`, { origin: base });
        bad.on("unexpected-response", (_request, response) => { resolve(response.statusCode); response.resume(); });
        bad.on("open", () => { bad.close(); reject(new Error("unauthenticated WebSocket accepted")); });
        bad.on("error", reject);
      });
      check(missingToken === 401, "Rust WebSocket requires a token");
      check((await fetch(`${base}/api/settings`)).status === 401,"Rust API requires a token");
      check((await fetch(`${base}/api/transcribe`,{method:"POST",body:"audio"})).status === 401 && (await fetch(`${base}/api/speak`,{method:"POST",body:"{}"})).status === 401,"Rust audio endpoints require a token");
      const snapshot = await fetch(`${base}/api/state`,{headers:{"x-liquid-token":token}}).then((res)=>res.json());
      check(snapshot.memory[0] === hello.memory[0] && snapshot.routines[0].id === hello.routines[0].id,"Rust HTTP and WebSocket snapshots agree");
      check(snapshot.parts.find((part)=>part.id === "small").status === "installed", "Rust server discovers installed models");
      check((await fetch(`${base}/api/settings`,{headers:{"x-liquid-token":token,Origin:"https://evil.example"}})).status === 401,"Rust API rejects foreign origins");
      const foreignHost = await new Promise((resolve, reject) => {
        const connection = net.connect(port,"127.0.0.1",() => connection.write(`GET /api/settings HTTP/1.1\r\nHost: evil.example\r\nX-Liquid-Token: ${token}\r\nConnection: close\r\n\r\n`));
        let result = ""; connection.on("data",(chunk) => result += chunk); connection.on("end",() => resolve(result)); connection.on("error",reject);
      });
      check(/^HTTP\/1\.1 401/.test(foreignHost),"Rust API rejects foreign hosts");
      check((await fetch(`${base}/escape.txt`)).status === 403,"Rust static server rejects symlink escapes");
      check((await fetch(`${base}/%2e%2e%2fsecret.txt`)).status === 403,"Rust static server rejects path traversal");
      const settingsEvent = once(socket,"message");
      const patch = await fetch(`${base}/api/settings`,{method:"PATCH",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({theme:"dark",guardianURL:"http://example.com"})}).then((res)=>res.json());
      check(patch.theme === "dark" && patch.guardianURL !== "http://example.com" && patch.guardianURL.startsWith("http://127.0.0.1"),"Rust API validates settings");
      const [settingsBytes] = await settingsEvent;
      check(JSON.parse(settingsBytes.toString()).settings.theme === "dark", "Rust WebSocket broadcasts settings changes");
      const api = async (route, method = "GET", body) => {
        const res = await fetch(`${base}${route}`, { method, headers: { "x-liquid-token":token, "content-type":"application/json" }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
        if (!res.ok) throw new Error(`${method} ${route} failed with ${res.status}`);
        return res.json();
      };
      const partEvent = new Promise((resolve, reject) => {
        const timer = setTimeout(()=>reject(new Error("model did not start")),15_000); timer.unref();
        const onMessage = (data) => { const event = JSON.parse(data.toString()); if (event.type === "parts" && event.parts.some((part)=>part.id === "small" && part.status === "running")) { clearTimeout(timer); socket.off("message",onMessage); resolve(event); } };
        socket.on("message",onMessage);
      });
      await api("/api/parts/small/start","POST");
      await partEvent;
      const stopState = (await api("/api/state")).parts.find((part)=>part.id === "small");
      check(stopState.status === "running","Rust server tracks model readiness");
      const finished = new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("text chat did not finish")),15_000); timer.unref();
        const onMessage=(data)=>{ const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.messages?.some((m)=>m.role === "assistant" && m.status === "done")) {clearTimeout(timer); socket.off("message",onMessage); resolve(event.chat);} };
        socket.on("message",onMessage);
      });
      check(stopState.context === 16384 && stopState.slots === 1 && stopState.capabilities.text === true, "Built-in text model reports a bounded context, one slot and text ability");
      const diagnostics=await api("/api/diagnostics");
      check(diagnostics.processes.server?.pid === child.pid && (process.platform !== "linux" || diagnostics.processes.server.pssBytes > 0), "Diagnostics includes the server process in memory samples");
      check(diagnostics.engineProperties.small?.state === "ready" && diagnostics.engineProperties.small?.device === "cpu", "Diagnostics reports the in-process model state and device");
      check((await fetch(`${base}/api/diagnostics`)).status === 401,"Diagnostics requires a token");
      const startedChat=await api("/api/chats","POST",{text:"Say hello"});
      const finishedChat=await finished;
      check(finishedChat.id === startedChat.id && finishedChat.messages.at(-1).content === "Hello friend!", "Rust server streams and saves a model reply");
      const modelRequest=lastRequest();
      check(modelRequest.stream === true && modelRequest.messages.some((m)=>m.role === "user" && m.content === "Say hello"),"Rust server sends chat context to the local model");
      // The models are told which language to answer in, and the prompt's words follow the language.
      check(modelRequest.messages.find((m)=>m.role === "system").content.includes("Reply in the same language as the user."),"Auto language tells the model to match the user's language");
      await api("/api/settings","PATCH",{language:"es"});
      const spanishDone=new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error("Spanish chat did not finish")),15_000); timer.unref(); const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type==="chat" && event.chat.messages?.at(-1)?.status==="done" && event.chat.messages.at(-1).content==="Hello friend!"){clearTimeout(timer); socket.off("message",onMessage); resolve(event.chat);}}; socket.on("message",onMessage);});
      const spanishChat=await api("/api/chats","POST",{text:"Say hello"});
      await spanishDone;
      const spanishSystem=lastRequest().messages.find((m)=>m.role === "system").content;
      check(spanishSystem.startsWith("Eres Liquid Agent") && spanishSystem.includes("Responde en Español."),"The models answer in the language the user chose, with prompts in that language");
      await api("/api/settings","PATCH",{language:"auto"});
      await api(`/api/chats/${spanishChat.id}`,"DELETE");
      const stopped = new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("chat did not stop")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.messages?.some((m)=>m.role === "assistant" && m.status === "stopped")) {clearTimeout(timer); socket.off("message",onMessage); resolve(event.chat);} };
        socket.on("message",onMessage);
      });
      const slowChat=await api("/api/chats","POST",{text:"please wait"});
      await api(`/api/chats/${slowChat.id}/stop`,"POST");
      check((await stopped).id === slowChat.id,"Rust server stops a streaming reply");
      const waitApproval = () => new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("tool did not request approval")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); const step=event.type === "chat" && event.chat.messages?.at(-1)?.steps?.find((s)=>s.status === "approval"); if(step){clearTimeout(timer);socket.off("message",onMessage);resolve({chat:event.chat,step});}};
        socket.on("message",onMessage);
      });
      const waitToolDone = () => new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("tool chat did not finish")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.messages?.at(-1)?.status === "done" && event.chat.messages.at(-1).steps?.length){clearTimeout(timer);socket.off("message",onMessage);resolve(event.chat);}};
        socket.on("message",onMessage);
      });
      if (chrome) {
      let lazyApproval=waitApproval(); const lazyDenied=await api("/api/chats","POST",{text:"browser test click"}); let lazyPending=await lazyApproval;
      let lazyDone=waitToolDone(); await api(`/api/approvals/${lazyPending.step.id}`,"POST",{approve:false}); await lazyDone;
      check((await api("/api/state")).parts.find(p=>p.id==="browser").status==="installed", "A denied browser action does not start the helper");
      await api(`/api/chats/${lazyDenied.id}`,"DELETE");
      lazyApproval=waitApproval(); const lazyApproved=await api("/api/chats","POST",{text:"browser test read"}); lazyPending=await lazyApproval;
      lazyDone=waitToolDone(); await api(`/api/approvals/${lazyPending.step.id}`,"POST",{approve:true}); await lazyDone;
      check((await api("/api/state")).parts.find(p=>p.id==="browser").status==="running", "An approved browser action starts the helper on demand");
      await api(`/api/chats/${lazyApproved.id}`,"DELETE");
      const bridge=async(action,body)=>(await fetch(`http://127.0.0.1:${browserPort}/${action}`,{method:"POST",headers:{"content-type":"application/json","x-liquid-bridge-token":token},body:JSON.stringify(body)})).json();
      check((await fetch(`http://127.0.0.1:${browserPort}/click`,{method:"POST",headers:{"content-type":"application/json"},body:'{"target":"Submit"}'})).status===401,"Rust-managed browser bridge rejects unauthenticated actions");
      const shop="data:text/html,<h1>Shop</h1><input placeholder=\"Search box\"><button onclick=\"document.querySelector('h1').innerText='Cli'+'cked'\">Submit</button>";
      check((await bridge("goto",{url:shop})).text?.includes("Submit"),"The Web helper opens a page and lists its buttons and fields");
      const loopback=await bridge("goto",{url:"http://127.0.0.1:12345/"});
      check(typeof loopback.error==="string" && loopback.error.includes("BLOCKED_BY_CLIENT"),"The Web helper refuses loopback addresses, whatever the agent asks");
      await bridge("goto",{url:shop}); // the blocked page above replaced the shop
      let browserApproval=waitApproval(); const browserChat=await api("/api/chats","POST",{text:"browser test click"}); let browserPending=await browserApproval;
      check(browserPending.step.tool==="browser_click" && !((await bridge("read",{})).text||"").includes("Clicked"),"Rust agent asks before a browser click");
      let browserDone=waitToolDone(); await api(`/api/approvals/${browserPending.step.id}`,"POST",{approve:true}); const browserReply=await browserDone;
      check(browserReply.messages.at(-1).steps[0].result.includes("Clicked"),"Rust agent sends approved browser action to the page");
      await api(`/api/chats/${browserChat.id}`,"DELETE");
      browserApproval=waitApproval(); const readBrowserChat=await api("/api/chats","POST",{text:"browser test read"}); browserPending=await browserApproval;
      browserDone=waitToolDone(); await api(`/api/approvals/${browserPending.step.id}`,"POST",{approve:true}); const readBrowserReply=await browserDone;
      check(readBrowserReply.messages.at(-1).steps[0].result.includes("Elements:"),"Rust agent reads the current browser page"); await api(`/api/chats/${readBrowserChat.id}`,"DELETE");
      browserApproval=waitApproval(); const deniedBrowserChat=await api("/api/chats","POST",{text:"browser test type"}); browserPending=await browserApproval;
      browserDone=waitToolDone(); await api(`/api/approvals/${browserPending.step.id}`,"POST",{approve:false}); const deniedBrowserReply=await browserDone;
      check(deniedBrowserReply.messages.at(-1).steps[0].status==="denied" && !((await bridge("read",{})).text||"").includes("hello"),"Denied browser typing never reaches the page");
      await api(`/api/chats/${deniedBrowserChat.id}`,"DELETE");
      browserApproval=waitApproval(); const failedBrowserChat=await api("/api/chats","POST",{text:"browser test fail"}); browserPending=await browserApproval;
      browserDone=waitToolDone(); await api(`/api/approvals/${browserPending.step.id}`,"POST",{approve:true}); const failedBrowserReply=await browserDone;
      check(failedBrowserReply.messages.at(-1).steps[0].status==="error" && failedBrowserReply.messages.at(-1).steps[0].result.includes("couldn't find"),"A missing element is reported as a failed browser action");
      await api(`/api/chats/${failedBrowserChat.id}`,"DELETE");
      browserApproval=waitApproval(); const localBrowserChat=await api("/api/chats","POST",{text:"browser test local"}); browserPending=await browserApproval;
      browserDone=waitToolDone(); await api(`/api/approvals/${browserPending.step.id}`,"POST",{approve:true}); const localBrowserReply=await browserDone;
      check(localBrowserReply.messages.at(-1).steps[0].status==="error" && localBrowserReply.messages.at(-1).steps[0].result.includes("public websites"),"Rust browser navigation rejects local addresses");
      await api(`/api/chats/${localBrowserChat.id}`,"DELETE");
      await api("/api/parts/browser/stop","POST");
      const afterStop=await api("/api/chats","POST",{text:"browser test read"});
      let stoppedReply;
      for(let i=0;i<150;i++){stoppedReply=(await api("/api/chats")).find(c=>c.id===afterStop.id)?.messages?.at(-1); if(stoppedReply?.status!=="streaming") break; await new Promise(resolve=>setTimeout(resolve,100));}
      check(stoppedReply?.status==="error" && (await api("/api/state")).parts.find(p=>p.id==="browser").status==="installed", "A manual browser stop suppresses later automatic startup");
      await api(`/api/chats/${afterStop.id}`,"DELETE");
      } else console.log("skipped the browser checks: no Chrome for Testing headless shell here (set LIQUID_TEST_CHROME)");
      const targetFile=path.join(dir,"agent-output.txt");
      let approvalEvent=waitApproval(); const deniedChat=await api("/api/chats","POST",{text:"Please write file"});
      let pending=await approvalEvent;
      check(pending.chat.id === deniedChat.id && !fs.existsSync(targetFile),"Rust agent asks before writing a file");
      let toolDone=waitToolDone(); await api(`/api/approvals/${pending.step.id}`,"POST",{approve:false});
      const denied=await toolDone;
      check(denied.messages.at(-1).steps[0].status === "denied" && !fs.existsSync(targetFile),"Rust agent does not execute denied actions");
      approvalEvent=waitApproval(); const allowedChat=await api("/api/chats","POST",{text:"Please write file"});
      pending=await approvalEvent; toolDone=waitToolDone();
      await api(`/api/approvals/${pending.step.id}`,"POST",{approve:true});
      const allowed=await toolDone;
      check(allowed.messages.at(-1).steps[0].status === "done" && fs.readFileSync(targetFile,"utf8") === "approved content","Rust agent runs approved file actions and continues the model response");
      fs.unlinkSync(targetFile);
      approvalEvent=waitApproval(); const webChat=await api("/api/chats","POST",{text:"Please fetch local page"});
      pending=await approvalEvent; check(pending.chat.id===webChat.id && pending.step.tool==="fetch_url","Rust web retrieval asks for approval in ask-always mode");
      toolDone=waitToolDone(); await api(`/api/approvals/${pending.step.id}`,"POST",{approve:true});
      const webReply=await toolDone;
      check(webReply.messages.at(-1).steps[0].status==="error" && webReply.messages.at(-1).steps[0].result.includes("public websites"),"Rust web retrieval blocks local addresses even after approval");
      await api(`/api/chats/${webChat.id}`,"DELETE");
      approvalEvent=waitApproval(); const cancelledChat=await api("/api/chats","POST",{text:"Please write file"});
      pending=await approvalEvent;
      const stoppedApproval = new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("approval did not cancel")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.id === cancelledChat.id && event.chat.messages?.at(-1)?.status === "stopped") {clearTimeout(timer);socket.off("message",onMessage);resolve(event.chat);} };
        socket.on("message",onMessage);
      });
      await api(`/api/chats/${cancelledChat.id}/stop`,"POST");
      const cancelled=await stoppedApproval;
      check(cancelled.messages.at(-1).steps[0].status === "denied" && !fs.existsSync(targetFile) && (await fetch(`${base}/api/approvals/${pending.step.id}`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({approve:true})})).status === 404,"Stopping a pending approval prevents execution and invalidates the decision");
      await api("/api/settings","PATCH",{approvalMode:"fullAuto"});
      approvalEvent=waitApproval(); const dangerChat=await api("/api/chats","POST",{text:"Run a dangerous shell command"});
      pending=await approvalEvent; toolDone=waitToolDone();
      check(pending.step.reason.includes("important parts"),"Dangerous commands require approval even in Full Auto");
      await api(`/api/approvals/${pending.step.id}`,"POST",{approve:false}); await toolDone;
      await api("/api/settings","PATCH",{approvalMode:"askAlways"});
      await api(`/api/chats/${dangerChat.id}`,"DELETE");
      await api("/api/settings","PATCH",{approvalMode:"fullAuto"});
      approvalEvent=waitApproval(); const delayedChat=await api("/api/chats","POST",{text:"Please do a delayed write file"});
      await api("/api/settings","PATCH",{approvalMode:"askAlways"});
      pending=await approvalEvent; toolDone=waitToolDone();
      check(pending.chat.id === delayedChat.id && !fs.existsSync(targetFile),"Approval uses the current setting when a model finishes generating");
      await api(`/api/approvals/${pending.step.id}`,"POST",{approve:false}); await toolDone;
      await api(`/api/chats/${delayedChat.id}`,"DELETE");
      const guardianReady = new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("guardian did not start")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "parts" && event.parts.some((p)=>p.id === "guardian" && p.status === "running")) {clearTimeout(timer);socket.off("message",onMessage);resolve();}};
        socket.on("message",onMessage);
      });
      await api("/api/settings","PATCH",{guardianURL:`http://127.0.0.1:${guardianPort}`,approvalMode:"approveForMe"});
      await guardianReady;
      for (let i=0;i<100;i++) { if ((await api("/api/settings")).approvalMode === "approveForMe") break; await new Promise((resolve)=>setTimeout(resolve,50)); if (i === 99) throw new Error("guardian mode did not activate"); }
      check((await api("/api/state")).guardianWanted === false,"Rust server starts the safety helper and enables the requested mode when ready");
      toolDone=waitToolDone(); const autoChat=await api("/api/chats","POST",{text:"Please write file"});
      const autoReply=await toolDone;
      check(autoReply.id === autoChat.id && autoReply.messages.at(-1).steps[0].decision === "Auto 150M approved" && autoReply.messages.at(-1).steps[0].pDeny === 0.1 && fs.existsSync(targetFile),"Rust agent uses a low safety score to approve an action");
      check((await api("/api/state")).activity.some((entry)=>entry.chatId === autoChat.id && entry.pDeny === 0.1 && entry.ok === true),"Rust agent records safety scores in activity");
      const safetyRequest=JSON.parse(fs.readFileSync(path.join(dir,"guardian-request.json"),"utf8"));
      const proposed=JSON.parse(safetyRequest.args);
      check(safetyRequest.tool === "Write" && safetyRequest.request === "Please write file" && proposed.file_path === targetFile && proposed.content === "approved content","Rust agent sends the expected request to the safety helper");
      fs.unlinkSync(targetFile); fs.writeFileSync(path.join(dir,"guardian-mode"),"high");
      approvalEvent=waitApproval(); const riskyChat=await api("/api/chats","POST",{text:"Please write file"});
      pending=await approvalEvent; toolDone=waitToolDone();
      check(pending.step.pDeny === 0.85 && pending.step.reason.includes("85%") && !fs.existsSync(targetFile),"Rust agent asks a person about a high-risk action");
      await api(`/api/approvals/${pending.step.id}`,"POST",{approve:false}); await toolDone;
      fs.writeFileSync(path.join(dir,"guardian-mode"),"invalid");
      approvalEvent=waitApproval(); const invalidChat=await api("/api/chats","POST",{text:"Please write file"});
      pending=await approvalEvent; toolDone=waitToolDone();
      check(pending.step.reason.includes("isn't ready") && !fs.existsSync(targetFile),"Invalid safety scores fall back to human approval");
      await api(`/api/approvals/${pending.step.id}`,"POST",{approve:false}); await toolDone;
      fs.writeFileSync(path.join(dir,"guardian-mode"),"delayed"); fs.unlinkSync(path.join(dir,"guardian-request.json"));
      approvalEvent=waitApproval(); const policyChat=await api("/api/chats","POST",{text:"Please write file"});
      for(let i=0;i<100 && !fs.existsSync(path.join(dir,"guardian-request.json"));i++) await new Promise(resolve=>setTimeout(resolve,10));
      check(fs.existsSync(path.join(dir,"guardian-request.json")),"Rust agent began the delayed safety check");
      await api("/api/settings","PATCH",{approvalMode:"askAlways"});
      pending=await approvalEvent; toolDone=waitToolDone();
      check(pending.chat.id === policyChat.id && pending.step.reason.includes("settings changed") && !fs.existsSync(targetFile),"A stricter approval setting takes effect during a safety check");
      await api(`/api/approvals/${pending.step.id}`,"POST",{approve:false}); await toolDone;
      await api("/api/settings","PATCH",{approvalMode:"approveForMe"});
      fs.writeFileSync(path.join(dir,"guardian-mode"),"slow"); fs.unlinkSync(path.join(dir,"guardian-request.json"));
      const checking = new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(new Error("safety check did not begin")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.messages?.at(-1)?.steps?.some((s)=>s.status === "checking")) {clearTimeout(timer);socket.off("message",onMessage);resolve(event.chat);} };
        socket.on("message",onMessage);
      });
      const checkingChat=await api("/api/chats","POST",{text:"Please write file"}); await checking;
      for(let i=0;i<100 && !fs.existsSync(path.join(dir,"guardian-request.json"));i++) await new Promise(resolve=>setTimeout(resolve,20));
      check(fs.existsSync(path.join(dir,"guardian-request.json")),"Rust agent sent the pending safety check");
      const stoppedChecking = new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(new Error("safety check did not stop")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.id === checkingChat.id && event.chat.messages?.at(-1)?.status === "stopped") {clearTimeout(timer);socket.off("message",onMessage);resolve(event.chat);} };
        socket.on("message",onMessage);
      });
      await api(`/api/chats/${checkingChat.id}/stop`,"POST"); const stoppedSafety=await stoppedChecking;
      check(stoppedSafety.messages.at(-1).steps[0].status === "denied" && !fs.existsSync(targetFile),"Stopping during safety classification prevents the action");
      await api("/api/parts/guardian/stop","POST");
      approvalEvent=waitApproval(); const offlineChat=await api("/api/chats","POST",{text:"Please write file"});
      pending=await approvalEvent; toolDone=waitToolDone();
      check(pending.step.reason.includes("isn't ready") && !fs.existsSync(targetFile),"An unavailable safety helper falls back to human approval");
      await api(`/api/approvals/${pending.step.id}`,"POST",{approve:false}); await toolDone;
      await api("/api/settings","PATCH",{approvalMode:"askAlways"});
      for(const chat of [autoChat,riskyChat,invalidChat,policyChat,offlineChat,checkingChat]) await api(`/api/chats/${chat.id}`,"DELETE");
      await api(`/api/chats/${deniedChat.id}`,"DELETE"); await api(`/api/chats/${allowedChat.id}`,"DELETE");
      await api(`/api/chats/${cancelledChat.id}`,"DELETE");
      const textUpload = await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":token,"x-file-name":"context.txt","content-type":"text/plain"},body:"The hidden detail is amber."}).then((res)=>res.json());
      const waitForReply = () => new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("attachment chat did not finish")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.messages?.at(-1)?.status === "done") {clearTimeout(timer); socket.off("message",onMessage); resolve(event.chat);} };
        socket.on("message",onMessage);
      });
      let responseEvent=waitForReply();
      const textChat=await api("/api/chats","POST",{text:"Read this",attachments:[textUpload.id]});
      check((await responseEvent).id === textChat.id,"Rust server replies to text attachments");
      let requestWithFiles=lastRequest();
      check(requestWithFiles.messages.some((m)=>typeof m.content === "string" && m.content.includes("The hidden detail is amber.")),"Rust server includes document text in model context");
      responseEvent=waitForReply();
      await api(`/api/chats/${textChat.id}/send`,"POST",{text:"What was that detail?"});
      check((await responseEvent).id === textChat.id,"Rust server continues chats that already contain attachments");
      requestWithFiles=lastRequest();
      check(requestWithFiles.messages.some((m)=>m.role === "user" && typeof m.content === "string" && m.content.includes("amber")) && requestWithFiles.messages.some((m)=>m.role === "user" && typeof m.content === "string" && m.content.includes("What was that detail?")),"Rust server retains prior attachment context for follow-ups");
      responseEvent=waitForReply();
      const editedMessageChat=await api(`/api/chats/${textChat.id}/edit`,"POST",{messageId:textChat.messages[0].id,text:"Read the updated question"});
      const editedResult=await responseEvent;
      check(editedResult.messages.length===2 && editedResult.messages[0].content==="Read the updated question" && editedResult.messages[0].id!==textChat.messages[0].id && editedResult.messages[0].attachments[0].id===textUpload.id,"Rust server edits a user message, preserves attachments, and replaces later replies");
      check(editedMessageChat.id===textChat.id && lastRequest().messages.some(m=>typeof m.content==="string" && m.content.includes("amber")),"Edited replies retain attachment context");
      responseEvent=waitForReply(); await api(`/api/chats/${textChat.id}/regenerate`,"POST"); const regenerated=await responseEvent;
      check(regenerated.messages.length===2 && regenerated.messages[0].id===editedResult.messages[0].id && regenerated.messages[1].id!==editedResult.messages[1].id,"Rust server regenerates the reply without duplicating the user message");
      const invalidEdit=await fetch(`${base}/api/chats/${textChat.id}/edit`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({messageId:"missing",text:"changed"})});
      check(invalidEdit.status===404,"Editing an unknown message is rejected");
      const pic=await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":token,"x-file-name":"vision.png","content-type":"image/png"},body:Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=","base64")}).then((res)=>res.json());
      const audio=await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":token,"x-file-name":"voice.wav","content-type":"audio/wav"},body:Buffer.from("RIFFtestWAVE")}).then((res)=>res.json());
      await api("/api/parts/omni/stop","POST");
      check((await fetch(`${base}/api/chats`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({text:"Describe",attachments:[pic.id]})})).status === 503,"Rust server requires the multimodal model for images");
      check((await fetch(`${base}/api/transcribe`,{method:"POST",headers:{"x-liquid-token":token},body:Buffer.from("RIFFtestWAVE")})).status === 503,"Rust dictation requires the multimodal model");
      await api("/api/settings","PATCH",{idleSleepMinutes:5,modelContext:8192});
      const omniReady = new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("multimodal model did not start")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "parts" && event.parts.some((p)=>p.id === "omni" && p.status === "running")) {clearTimeout(timer); socket.off("message",onMessage); resolve();} };
        socket.on("message",onMessage);
      });
      await api("/api/parts/omni/start","POST"); await omniReady;
      const omniState=(await api("/api/state")).parts.find((p)=>p.id==="omni");
      check(omniState.context===8192 && omniState.slots===1 && omniState.capabilities.vision===true, "Omni uses the selected 8K context, one slot and sees pictures");
      check((await api("/api/settings")).idleSleepMinutes===5, "Selected idle sleep is saved");
      check((await fetch(`${base}/api/transcribe`,{method:"POST",body:"audio"})).status === 401 && (await fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":token,Origin:"https://evil.example"},body:"{}"})).status === 401,"Rust audio endpoints remain protected while the model is running");
      check((await fetch(`${base}/api/capabilities/omni`,{method:"POST"})).status === 401,"Capability check requires a token");
      const capability=await api("/api/capabilities/omni","POST");
      check(capability.spanishWav === true && capability.asr === true && capability.vision === true && capability.voiceSelection === false && capability.speakerReference === false && capability.realtimeFullDuplex === false, "Capability check runs real speech, transcription and picture operations and reports unsupported voice controls");
      responseEvent=waitForReply();
      const mediaChat=await api("/api/chats","POST",{text:"",attachments:[pic.id,audio.id]});
      check((await responseEvent).id === mediaChat.id && mediaChat.title === "vision.png","Rust server replies to attachment-only image and audio messages");
      requestWithFiles=lastRequest();
      const mediaParts=requestWithFiles.messages.flatMap((m)=>Array.isArray(m.content)?m.content:[]);
      check(mediaParts.some((p)=>p.type === "image_url" && p.image_url.url.startsWith("data:image/png;base64,")) && mediaParts.some((p)=>p.type === "input_audio" && p.input_audio.format === "wav"),"Rust server routes image and audio bytes to the multimodal model");
      const transcript=await fetch(`${base}/api/transcribe`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"audio/wav"},body:Buffer.from("RIFFtestWAVE")}).then((res)=>res.json());
      check(transcript.text === "hello from the test","Rust server transcribes audio through the local model");
      const asrRequest=lastRequest();
      check(asrRequest.messages[0].content === "Perform ASR." && asrRequest.messages[1].content[0].input_audio.format === "wav","Rust server sends WAV data to the dictation model");
      await api("/api/settings","PATCH",{asrLanguage:"es"});
      await fetch(`${base}/api/transcribe`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"audio/wav"},body:Buffer.from("RIFFtestWAVE")});
      const spanishAsr=lastRequest();
      check(spanishAsr.messages[0].content === "Perform ASR in Spanish.","Dictation uses its independently selected language");
      await api("/api/settings","PATCH",{asrLanguage:"auto"});
      const audioRequests=() => fs.existsSync(path.join(dir,"model-request.jsonl")) ? fs.readFileSync(path.join(dir,"model-request.jsonl"),"utf8").trim().split("\n").filter((line)=>line.includes("input_audio")).length : 0;
      const beforeSpeech=audioRequests();
      const speechResponse=await fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({text:"Hola",voice:"male",lang:"es"})});
      const speech=Buffer.from(await speechResponse.arrayBuffer());
      check(speechResponse.headers.get("content-type") === "audio/wav" && speech.toString("ascii",0,4) === "RIFF" && speechResponse.headers.get("x-liquid-voice-applied") === "false","Rust server speaks with the in-process model and reports that the voice choice is not applied");
      check(audioRequests() === beforeSpeech,"Default speech does not run a transcription check");
      await api("/api/settings","PATCH",{verifySpeech:true});
      const checked=await fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({text:"Hello friend!",voice:"female"})});
      await checked.arrayBuffer();
      check(checked.ok && audioRequests() > beforeSpeech,"Optional speech verification transcribes the clip it made");
      await api("/api/settings","PATCH",{verifySpeech:false});
      const invalidLang=await fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({text:"Hola",lang:"invalid"})});
      check(invalidLang.status === 400,"Speech rejects an unsupported language override");
      await api("/api/settings","PATCH",{ttsLanguage:"auto",verifySpeech:false});
      const benchmark=spawn(process.execPath,[path.join(repo,"scripts/benchmark-models.mjs"),"--url",base,"--mode","both","--turns","2","--interval-ms","250"],{stdio:["ignore","pipe","pipe"]});
      let benchmarkOutput="",benchmarkError="";
      benchmark.stdout.on("data",chunk=>benchmarkOutput+=chunk); benchmark.stderr.on("data",chunk=>benchmarkError+=chunk);
      const [benchmarkCode]=await once(benchmark,"exit");
      check(benchmarkCode===0,`Benchmark completed against fake engines: ${benchmarkError}`);
      const report=JSON.parse(benchmarkOutput);
      check(report.status === "completed" && report.latenciesMs.chatCompleteMs.count===2 && report.latenciesMs.ttsCompleteMs.count===2 && report.samples.length>0 && report.limitations.some(s=>s.includes("complete-clip")),"Benchmark reports repeated operation latency and sampled resource limits");
      check(report.latenciesMs.firstTextMs.count===2 && report.latenciesMs.firstTextMs.median < report.latenciesMs.chatCompleteMs.median,"Benchmark measures first streamed text before the completed reply");
      check(process.platform !== "linux" || (report.sampledManagedPssBytes?.max > 0 && report.samples.some(s=>s.processes?.server?.pssBytes>0)),"Benchmark samples the server's memory, which includes the in-process models");
      const audioFixture=path.join(dir,"benchmark.wav");fs.writeFileSync(audioFixture,"RIFFtestWAVE");
      const asrBenchmark=spawn(process.execPath,[path.join(repo,"scripts/benchmark-models.mjs"),"--url",base,"--mode","asr","--audio",audioFixture,"--turns","2","--interval-ms","250"],{stdio:["ignore","pipe","pipe"]});
      let asrOutput="";asrBenchmark.stdout.on("data",chunk=>asrOutput+=chunk);const [asrCode]=await once(asrBenchmark,"exit");
      const asrReport=JSON.parse(asrOutput);
      check(asrCode===0 && asrReport.latenciesMs.asrCompleteMs.count===2 && asrReport.transcriptCharacters.count===2 && asrReport.audioInputBytes===12 && asrReport.benchmarkChatId===null,"Benchmark measures repeated dictation without creating a chat");
      const parallel=spawn(process.execPath,[path.join(repo,"scripts/benchmark-models.mjs"),"--url",base,"--mode","concurrent","--turns","2","--interval-ms","250"],{stdio:["ignore","pipe","pipe"]});
      let parallelOutput="";parallel.stdout.on("data",chunk=>parallelOutput+=chunk);const [parallelCode]=await once(parallel,"exit");
      const parallelReport=JSON.parse(parallelOutput);
      check(parallelCode===0 && parallelReport.latenciesMs.firstTextMs.count===2 && parallelReport.latenciesMs.ttsCompleteMs.count===2,"Concurrent benchmark streams chat and speech for each turn");
      const oversizedAudio=path.join(dir,"oversized-benchmark.wav");const audioFd=fs.openSync(oversizedAudio,"w");fs.ftruncateSync(audioFd,25*1024*1024+1);fs.closeSync(audioFd);
      const tooLarge=spawn(process.execPath,[path.join(repo,"scripts/benchmark-models.mjs"),"--mode","asr","--audio",oversizedAudio],{stdio:["ignore","ignore","pipe"]});
      let sizeError="";tooLarge.stderr.on("data",chunk=>sizeError+=chunk);const [sizeCode]=await once(tooLarge,"exit");
      check(sizeCode!==0 && sizeError.includes("25 MiB"),"ASR benchmark rejects an oversized input before loading it");
      await api("/api/parts/omni/stop","POST");
      await api(`/api/chats/${textChat.id}`,"DELETE");
      await api(`/api/chats/${mediaChat.id}`,"DELETE");
      const toolError = new Promise((resolve, reject) => {
        const timer=setTimeout(()=>reject(new Error("unsupported tool call did not fail")),15_000); timer.unref();
        const onMessage=(data)=>{const event=JSON.parse(data.toString()); if(event.type === "chat" && event.chat.messages?.some((m)=>m.role === "assistant" && m.status === "error")) {clearTimeout(timer); socket.off("message",onMessage); resolve(event.chat);} };
        socket.on("message",onMessage);
      });
      const toolChat=await api("/api/chats","POST",{text:"try tool"});
      const failedChat=await toolError;
      check(failedChat.id === toolChat.id && !failedChat.messages.at(-1).content.includes("<tool") && failedChat.messages.at(-1).error.includes("Tool actions"),"Rust server does not present unsupported tool calls as completed actions");
      await api(`/api/chats/${startedChat.id}`,"DELETE");
      await api(`/api/chats/${slowChat.id}`,"DELETE");
      await api(`/api/chats/${toolChat.id}`,"DELETE");
      await api("/api/parts/small/stop","POST");
      check((await api("/api/state")).parts.find((part)=>part.id === "small").status === "installed","Rust server stops its model process");
      check((await api("/api/memory"))[0] === "Already saved fact.","Rust server reads existing memory");
      const memoryEvent = once(socket,"message");
      check((await api("/api/memory","POST",{fact:"  New   fact.  "}))[1] === "New fact.","Rust server adds normalized memory");
      const [memoryBytes] = await memoryEvent;
      check(JSON.parse(memoryBytes.toString()).facts.includes("New fact."), "Rust WebSocket broadcasts memory changes");
      check((await api("/api/routines"))[0].id === "existing","Rust server reads existing routines");
      const routineEvent = once(socket,"message");
      const routine = await api("/api/routines","POST",{name:" New ",prompt:"Do something",schedule:{enabled:true,time:"08:30",days:[5,1,1,8]}});
      check(routine.name === "New" && routine.schedule.days.join(",") === "1,5","Rust server creates validated routines");
      const [routineBytes] = await routineEvent;
      check(JSON.parse(routineBytes.toString()).routines.some((r)=>r.id === routine.id), "Rust WebSocket broadcasts routine changes");
      const edited = await api(`/api/routines/${routine.id}`,"PATCH",{schedule:{time:"bad",days:[3]}});
      check(edited.schedule.time === "08:30" && !edited.schedule.enabled,"Rust server updates routines");
      const newChatEvent = new Promise((resolve) => {
        const onMessage = (data) => { const event = JSON.parse(data.toString()); if (event.type === "chat") { socket.off("message", onMessage); resolve(event); } };
        socket.on("message", onMessage);
      });
      const chat = await api("/api/chats","POST",{});
      const chatEvent = await newChatEvent;
      check(chat.title === "New chat" && chatEvent.chat.id === chat.id,"Rust server creates and broadcasts chats");
      const editedChat = await api(`/api/chats/${chat.id}`,"PATCH",{title:"  Named chat  ",pinned:true});
      check(editedChat.title === "Named chat" && editedChat.pinned && editedChat.named,"Rust server updates chat metadata");
      check((await fetch(`${base}/api/chats`,{method:"POST",headers:{"x-liquid-token":token,"content-type":"application/json"},body:JSON.stringify({text:"hello"})})).status === 503,"Rust server reports unavailable chat execution");
      check(await fetch(`${base}/api/uploads/file?t=${token}`).then((res)=>res.text()) === "old attachment","Rust server serves existing attachments");
      check((await fetch(`${base}/api/uploads/file`)).status === 401,"Rust server protects attachment access");
      check((await fetch(`${base}/api/uploads/unsafe?t=${token}`)).status === 404,"Rust server refuses attachments outside its data directory");
      check((await fetch(`${base}/api/chats/unsafechat`,{method:"DELETE",headers:{"x-liquid-token":token}})).status === 409 && fs.existsSync(outsideFile),"Rust server refuses unsafe attachment deletion");
      const upload = await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":token,"x-file-name":encodeURIComponent("notes.txt"),"content-type":"text/plain"},body:"hello attachment"}).then((res)=>res.json());
      check(upload.name === "notes.txt" && upload.kind === "text" && await fetch(`${base}/api/uploads/${upload.id}?t=${token}`).then((res)=>res.text()) === "hello attachment","Rust server stores and serves text attachments");
      check(fs.readFileSync(path.join(data,"uploads",`${upload.id}.txt.txt`),"utf8") === "hello attachment","Rust server writes text extraction sidecars");
      const image = await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":token,"x-file-name":"picture.png","content-type":"image/png"},body:Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=","base64")}).then((res)=>res.json());
      check(image.kind === "image" && image.mime === "image/png","Rust server preserves supported image uploads");
      if (process.platform === "darwin" || ["magick", "convert", "ffmpeg"].some((command)=>spawnSync(command,["-version"],{stdio:"ignore"}).status === 0)) {
        const bmp = Buffer.alloc(58); bmp.write("BM"); bmp.writeUInt32LE(58,2); bmp.writeUInt32LE(54,10); bmp.writeUInt32LE(40,14); bmp.writeInt32LE(1,18); bmp.writeInt32LE(1,22); bmp.writeUInt16LE(1,26); bmp.writeUInt16LE(24,28); bmp.writeUInt32LE(4,34); bmp[56]=255;
        const converted = await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":token,"x-file-name":"picture.bmp","content-type":"image/bmp"},body:bmp}).then((res)=>res.json());
        const convertedBytes = Buffer.from(await fetch(`${base}/api/uploads/${converted.id}?t=${token}`).then((res)=>res.arrayBuffer()));
        check(converted.mime === "image/png" && convertedBytes.subarray(0,8).equals(Buffer.from("89504e470d0a1a0a","hex")),"Rust server converts BMP images to PNG");
      }
      const oversized = await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":token,"x-file-name":"large.bin","content-type":"application/octet-stream"},body:Buffer.alloc(25*1024*1024+1)});
      check(oversized.status === 413,"Rust server enforces the attachment size limit");
      await api("/api/settings","PATCH",{startupMode:"single-omni"});
      socket.close(); await once(socket,"close");
      await stop(); env.LIQUID_AGENT_NO_AUTOSTART="0"; await start();
      const nextPage = await (await fetch(base)).text(); const nextToken = nextPage.match(/name="liquid-token" content="([^"]+)"/)?.[1];
      check(nextToken && nextToken !== token,"Rust server rotates tokens at restart");
      const startupParts=(await fetch(`${base}/api/state`,{headers:{"x-liquid-token":nextToken}}).then(r=>r.json())).parts;
      check(startupParts.find(p=>p.id==="omni")?.status==="installed" && startupParts.find(p=>p.id==="omni")?.onDemand === true && startupParts.find(p=>p.id==="small")?.status==="installed" && startupParts.find(p=>p.id==="guardian")?.status==="installed", "Single-Omni mode leaves models unloaded and offers on-demand startup");
      await fetch(`${base}/api/parts/install`,{method:"POST",headers:{"x-liquid-token":nextToken,"content-type":"application/json"},body:JSON.stringify({ids:["small","omni"]})});
      let postInstall;
      for(let i=0;i<100;i++){postInstall=await fetch(`${base}/api/state`,{headers:{"x-liquid-token":nextToken}}).then(r=>r.json()); if(!postInstall.busy)break; await new Promise(resolve=>setTimeout(resolve,50));}
      check(!postInstall.busy && postInstall.parts.find(p=>p.id==="small")?.status==="installed" && postInstall.parts.find(p=>p.id==="omni")?.status==="installed", "Single-Omni installation leaves both models unloaded");
      const badSpeech=await fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":nextToken,"content-type":"application/json"},body:JSON.stringify({text:"Hola",lang:"invalid"})});
      const emptySpeech=await fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":nextToken,"content-type":"application/json"},body:JSON.stringify({text:"  "})});
      const emptyAudio=await fetch(`${base}/api/transcribe`,{method:"POST",headers:{"x-liquid-token":nextToken},body:""});
      check(badSpeech.status===400 && emptySpeech.status===400 && emptyAudio.status===400,"Invalid or empty audio requests are refused");
      const [lowChat, coldSpeech]=await Promise.all([
        fetch(`${base}/api/chats`,{method:"POST",headers:{"x-liquid-token":nextToken,"content-type":"application/json"},body:JSON.stringify({text:"Low memory chat"})}).then(r=>r.json()),
        fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":nextToken,"content-type":"application/json"},body:JSON.stringify({text:"Hola",lang:"es"})})
      ]);
      check(coldSpeech.ok,"Concurrent chat and speech share one on-demand Omni startup");
      let lowResult;
      for (let i=0;i<150;i++) {
        lowResult=(await fetch(`${base}/api/chats`,{headers:{"x-liquid-token":nextToken}}).then(r=>r.json())).find(c=>c.id===lowChat.id);
        if (lowResult?.messages?.at(-1)?.status === "done") break;
        await new Promise(resolve=>setTimeout(resolve,100));
      }
      check(lowResult?.messages?.at(-1)?.status === "done" && lowResult.messages.at(-1).model === "omni", "Low-memory startup routes text chat to Omni");
      await fetch(`${base}/api/chats/${lowChat.id}`,{method:"DELETE",headers:{"x-liquid-token":nextToken}});
      await fetch(`${base}/api/parts/omni/stop`,{method:"POST",headers:{"x-liquid-token":nextToken}});
      const stoppedParts=(await fetch(`${base}/api/state`,{headers:{"x-liquid-token":nextToken}}).then(r=>r.json())).parts;
      const blockedWake=await fetch(`${base}/api/speak`,{method:"POST",headers:{"x-liquid-token":nextToken,"content-type":"application/json"},body:JSON.stringify({text:"Hola"})});
      check(stoppedParts.find(p=>p.id==="omni")?.onDemand === false && blockedWake.status===503,"Manual stop prevents on-demand restart");
      const manualWake=await fetch(`${base}/api/parts/omni/start`,{method:"POST",headers:{"x-liquid-token":nextToken}});
      check(manualWake.ok,"Manual start re-enables Omni after an explicit stop");
      const override=await fetch(`${base}/api/parts/small/start`,{method:"POST",headers:{"x-liquid-token":nextToken}});
      check(override.ok,"An explicit manual model start can override low-memory startup");

      const saved = await fetch(`${base}/api/settings`,{headers:{"x-liquid-token":nextToken}}).then((res)=>res.json());
      check(saved.theme === "dark","Rust server persists settings");
      const memory = await fetch(`${base}/api/memory`,{headers:{"x-liquid-token":nextToken}}).then((res)=>res.json());
      const routines = await fetch(`${base}/api/routines`,{headers:{"x-liquid-token":nextToken}}).then((res)=>res.json());
      check(memory.includes("New fact.") && routines.some((r)=>r.id === routine.id && r.schedule.days[0] === 3),"Rust server persists memory and routines");
      const chats = await fetch(`${base}/api/chats`,{headers:{"x-liquid-token":nextToken}}).then((res)=>res.json());
      check(chats.some((c)=>c.id === chat.id && c.title === "Named chat") && chats.some((c)=>c.id === "oldchat"),"Rust server persists chat metadata");
      await fetch(`${base}/api/chats/${chat.id}`,{method:"DELETE",headers:{"x-liquid-token":nextToken}});
      const remainingChats = await fetch(`${base}/api/chats`,{headers:{"x-liquid-token":nextToken}}).then((res)=>res.json());
      check(remainingChats.length === 3 && remainingChats.some((c)=>c.id === "oldchat"),"Rust server deletes an attachment-free chat");
      const oldDelete = await fetch(`${base}/api/chats/oldchat`,{method:"DELETE",headers:{"x-liquid-token":nextToken}});
      check(oldDelete.ok && fs.existsSync(existingFile) && (await fetch(`${base}/api/uploads/file?t=${nextToken}`)).status === 200,"Rust server preserves attachments referenced by another chat");
      const sharedDelete = await fetch(`${base}/api/chats/sharedchat`,{method:"DELETE",headers:{"x-liquid-token":nextToken}});
      check(sharedDelete.ok && !fs.existsSync(existingFile) && (await fetch(`${base}/api/uploads/file?t=${nextToken}`)).status === 404,"Rust server removes unreferenced chat attachments");
      await fetch(`${base}/api/memory/1`,{method:"DELETE",headers:{"x-liquid-token":nextToken}});
      await fetch(`${base}/api/routines/${routine.id}`,{method:"DELETE",headers:{"x-liquid-token":nextToken}});
      const remainingFacts = await fetch(`${base}/api/memory`,{headers:{"x-liquid-token":nextToken}}).then((res)=>res.json());
      const remainingRoutines = await fetch(`${base}/api/routines`,{headers:{"x-liquid-token":nextToken}}).then((res)=>res.json());
      check(remainingFacts.length === 1 && remainingFacts[0] === "Already saved fact." && remainingRoutines.length === 1 && remainingRoutines[0].id === "existing","Rust server removes selected memory and routines");
      await fetch(`${base}/api/settings`,{method:"PATCH",headers:{"x-liquid-token":nextToken,"content-type":"application/json"},body:JSON.stringify({startupMode:"hybrid"})});
      await stop(); env.LIQUID_AGENT_HYBRID_IDLE_SECONDS="2"; await start();
      const hybridPage=await (await fetch(base)).text(); const hybridToken=hybridPage.match(/name="liquid-token" content="([^"]+)"/)?.[1];
      const hybridHeaders={"x-liquid-token":hybridToken,"content-type":"application/json"};
      const hybridState=async()=>await fetch(`${base}/api/state`,{headers:hybridHeaders}).then(r=>r.json());
      async function hybridWait(test) { let last; for(let i=0;i<200;i++){last=await hybridState(); if(test(last)) return last; await new Promise(resolve=>setTimeout(resolve,100));} throw Error("Hybrid transition timed out: "+JSON.stringify(last.parts.filter(p=>["small","omni"].includes(p.id)).map(p=>({id:p.id,status:p.status,error:p.error})))); }
      let hybrid=await hybridWait(s=>s.parts.find(p=>p.id==="small")?.status==="running");
      check(hybrid.parts.find(p=>p.id==="omni")?.onDemand===true,"Hybrid mode starts text and offers Omni on demand");
      await fetch(`${base}/api/settings`,{method:"PATCH",headers:hybridHeaders,body:JSON.stringify({startupMode:"auto"})});
      const autoState=await hybridState();
      check(autoState.parts.find(p=>p.id==="omni")?.onDemand===true,"Automatic mode offers installed Omni when text is running");
      const autoSpeech=await fetch(`${base}/api/speak`,{method:"POST",headers:hybridHeaders,body:JSON.stringify({text:"Hola",lang:"es"})});
      check(autoSpeech.ok,"Automatic mode switches to Omni on demand for speech");
      await fetch(`${base}/api/settings`,{method:"PATCH",headers:hybridHeaders,body:JSON.stringify({startupMode:"hybrid"})});
      const hybridImage=await fetch(`${base}/api/uploads`,{method:"POST",headers:{"x-liquid-token":hybridToken,"x-file-name":"hybrid.png","content-type":"image/png"},body:Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=","base64")}).then(r=>r.json());
      const imageChat=await fetch(`${base}/api/chats`,{method:"POST",headers:hybridHeaders,body:JSON.stringify({text:"Describe this picture",attachments:[hybridImage.id]})}).then(r=>r.json());
      hybrid=await hybridWait(s=>s.chats.find(c=>c.id===imageChat.id)?.messages.at(-1)?.status==="done");
      check(hybrid.chats.find(c=>c.id===imageChat.id)?.messages.at(-1)?.model==="omni", "Hybrid warms Omni for an image attachment");
      await fetch(`${base}/api/chats/${imageChat.id}`,{method:"DELETE",headers:hybridHeaders});

    } finally { socket?.terminate(); if (child) await stop(); fs.rmSync(dir,{recursive:true,force:true}); }
  }, 600_000);
});
