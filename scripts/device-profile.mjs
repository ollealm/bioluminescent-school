#!/usr/bin/env node
/**
 * Run `__stages()` (src/gpu/profile.ts) on a cabled iPhone's Safari and
 * print where its GPU time goes, at one or more fish counts. No manual
 * tapping: macOS's safaridriver drives the phone over WebDriver.
 *
 * Once per phone: plug it in and trust this Mac, then on the phone turn on
 * Settings → Apps → Safari → Advanced → Web Inspector and Remote Automation.
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
process.on("SIGINT", () => process.exit(130));

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

const session = await call("POST", "/session", {
  capabilities: {
    alwaysMatch: {
      browserName: "Safari",
      platformName: "iOS",
      ...(args.udid ? { "safari:deviceUDID": args.udid } : {}),
    },
  },
});
const sid = session.sessionId;
const s = (path) => `/session/${sid}${path}`;

try {
  await call("POST", s("/timeouts"), { script: 180_000, pageLoad: 60_000 });
  const ua = await call("POST", s("/execute/sync"), { script: "return navigator.userAgent", args: [] });
  console.log(`device: ${ua}\n`);
  for (const count of counts) {
    // governOff: the count stays put while we measure.
    await call("POST", s("/url"), { url: `${base}/?count=${count}&governOff&diag` });
    await sleep(4000); // boot, compile, settle
    const report = await call("POST", s("/execute/async"), {
      script: `const done = arguments[arguments.length - 1];
        if (!window.__stages) return done({ error: "no __stages() on this page" });
        __stages(arguments[0]).then(done, (e) => done({ error: String(e) }));`,
      args: [{ repeat, samples }],
    });
    if (report.error) {
      console.log(`${count} fish: ${report.error}`);
      continue;
    }
    console.log(
      `${report.count} fish · ${report.pixels.join("×")} px · frame ${report.frameMs} ms ` +
        `(empty submit ${report.emptySubmitMs}, stages sum ${report.sumMs})`,
    );
    const rows = Object.entries(report.stagesMs).sort((a, b) => b[1] - a[1]);
    for (const [name, ms] of rows) console.log(`  ${name.padEnd(9)} ${ms.toFixed(2).padStart(6)} ms`);
    console.log();
  }
} finally {
  await call("DELETE", s("")).catch(() => {});
}
process.exit(0);
