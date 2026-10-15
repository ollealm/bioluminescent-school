#!/bin/sh
# Run a CPU-engine dev tool (see each .ts file's header):
#   sh scripts/cpu/run.sh bench | determinism
set -e
cd "$(dirname "$0")/../.."
tool="$1"
out="$(mktemp -d)/$tool.mjs"
# The benchmark also runs in jsc (no Node APIs); the determinism check hashes with node:crypto.
platform=neutral; [ "$tool" = determinism ] && platform=node
node_modules/.bin/esbuild "scripts/cpu/$tool.ts" --bundle --format=esm --platform="$platform" \
  --main-fields=module,main --log-level=warning --outfile="$out" \
  --alias:three="$PWD/node_modules/three/build/three.module.js"
if [ "$tool" = bench ]; then
  echo "V8 (node):"; node "$out"
  JSC=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
  if [ -x "$JSC" ]; then echo; echo "JavaScriptCore (jsc):"; "$JSC" -m "$out"; fi
else
  node "$out"
fi
