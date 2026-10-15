#!/usr/bin/env node
/**
 * Run `__stages()` (src/gpu/profile.ts) on a cabled iPhone's Safari and
 * print where its GPU time goes, at one or more fish counts. No manual
 * tapping: macOS's safaridriver drives the phone over WebDriver.
 *
 * Once per phone: plug it in and trust this Mac, then on the phone turn on
 * Settings → Apps → Safari → Advanced → Web Inspector and Remote Automation,
 * and set Settings → Display & Brightness → Auto-Lock → Never (a locked
 * phone refuses the session, and only the phone can unlock itself).
 *
 *   node scripts/device-profile.mjs [--url https://olle.work] [--counts 1000,12000]
 *                                   [--udid <id>] [--repeat 10] [--samples 7]
 *
 * The page must be served over HTTPS (WebGPU needs a secure context) and
 * carry this repo's __stages(). `xcrun devicectl list devices` lists UDIDs;
 * without --udid safaridriver picks the connected device.
 */
import { spawn } from "node:child_process";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, a, i, all) => {
    if (a.startsWith("--")) pairs.push([a.slice(2), all[i + 1]]);
    return pairs;
  }, []),
);
const base = args.url ?? "https://olle.work";
const counts = (args.counts ?? "1000,12000").split(",").map(Number);
const repeat = Number(args.repeat ?? 10);
const samples = Number(args.samples ?? 7);
const PORT = 4723;
const wd = `http://localhost:${PORT}`;

const driver = spawn("safaridriver", ["-p", String(PORT)], { stdio: "inherit" });
const stop = () => driver.kill();
process.on("exit", stop);

async function call(method, path, body) {
  const res = await fetch(wd + path, {
    method,
    headers: { "content-type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${method} ${path}: ${JSON.stringify(json.value ?? json)}`);
  return json.value;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// safaridriver takes a moment to listen.
for (let i = 0; ; i++) {
  try {
    await fetch(`${wd}/status`);
    break;
  } catch {
    if (i > 50) throw new Error("safaridriver didn't start");
    await sleep(100);
  }
}

// A locked (or just re-trusted) phone refuses the session: wait up to two
// minutes for it, so the phone can be unlocked while the script waits.
let session;
for (let waited = 0; ; waited += 3) {
  try {
    session = await call("POST", "/session", {
      capabilities: {
        alwaysMatch: {
          browserName: "Safari",
          platformName: "iOS",
          ...(args.udid ? { "safari:deviceUDID": args.udid } : {}),
        },
      },
    });
    break;
  } catch (err) {
    if (!/locked|not paired|could not be used/.test(String(err)) || waited >= 120) throw err;
    if (waited === 0) console.log("waiting for the phone (unlock it, accept Trust if asked)...");
    await sleep(3000);
  }
}
const sid = session.sessionId;
const s = (path) => `/session/${sid}${path}`;
// Stopped mid-run: end the session first, or the phone's Safari stays held
// by it ("already paired with a different session") until Safari is quit.
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => {
    call("DELETE", s("")).catch(() => {}).finally(() => process.exit(130));
  });
}

try {
  await call("POST", s("/timeouts"), { script: 180_000, pageLoad: 60_000 });
  const ua = await call("POST", s("/execute/sync"), { script: "return navigator.userAgent", args: [] });
  console.log(`device: ${ua}\n`);
  for (const count of counts) {
    // governOff: the count stays put while we measure.
    await call("POST", s("/url"), { url: `${base}/?count=${count}&governOff&diag` });
    const report = await call("POST", s("/execute/async"), {
      // Wait for the page to boot (a fresh URL on a phone can take a while),
      // then let it settle before timing.
      script: `const done = arguments[arguments.length - 1];
        const opts = arguments[0];
        const t0 = Date.now();
        // Keep the screen on for the run (Screen Wake Lock); Auto-Lock still
        // applies between runs.
        navigator.wakeLock?.request("screen").catch(() => {});
        (function wait() {
          if (!window.__stages || !window.swarm) {
            if (Date.now() - t0 > 30000) {
              return done({
                error: "page not ready after 30 s",
                page: {
                  href: location.href,
                  title: document.title,
                  stages: typeof window.__stages,
                  swarm: typeof window.swarm,
                  webgpu: "gpu" in navigator,
                  text: document.body?.innerText.slice(0, 300),
                },
              });
            }
            return setTimeout(wait, 250);
          }
          setTimeout(() => __stages(opts).then(done, (e) => done({ error: String(e) })), 3000);
        })();`,
      args: [{ repeat, samples }],
    });
    if (report.error) {
      console.log(`${count} fish: ${report.error}`, report.page ? JSON.stringify(report.page) : "");
      continue;
    }
    console.log(
      `${report.count} fish · ${report.pixels.join("×")} px · frame ${report.frameMs} ms ` +
        `(js ${report.cpuMs}, empty submit ${report.emptySubmitMs}, stages sum ${report.sumMs})`,
    );
    const rows = Object.entries(report.stagesMs).sort((a, b) => b[1] - a[1]);
    for (const [name, ms] of rows) console.log(`  ${name.padEnd(9)} ${ms.toFixed(2).padStart(6)} ms`);
    console.log();
  }
} finally {
  await call("DELETE", s("")).catch(() => {});
}
process.exit(0);
