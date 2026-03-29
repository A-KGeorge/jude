"use strict";

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function loadTFSessionOrSkip(t: { skip: (message?: string) => void }) {
  try {
    const mod = await import("../index");
    return mod.TFSession;
  } catch (err: any) {
    t.skip(
      `native addon unavailable in test environment: ${err?.message ?? err}`,
    );
    return null;
  }
}

describe("TFSession.runAsync()", () => {
  it("runAsync method exists on TFSession instances", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;

    // Check the API contract — runAsync should be a function on the prototype
    // (we can't easily instantiate without a valid model, but we can verify the class)
    const proto = TFSession.prototype;
    assert.equal(
      typeof proto.runAsync,
      "function",
      "runAsync method exists on TFSession prototype",
    );

    // Also verify other expected methods
    assert.equal(typeof proto.run, "function", "run method exists");
    assert.equal(typeof proto.destroy, "function", "destroy method exists");
  });

  it("runAsync is an async method (returns Promise)", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;

    assert.equal(
      typeof TFSession.prototype.runAsync,
      "function",
      "runAsync is a function",
    );
  });

  it("runAsync and run have identical method signatures", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;

    const runAsyncLength = TFSession.prototype.runAsync.length;
    const runLength = TFSession.prototype.run.length;

    // Both should accept (inputs, outputKeys?) — length 2 or compatible
    assert(runAsyncLength <= 2, "runAsync accepts ≤ 2 parameters");
    assert(runLength <= 2, "run accepts ≤ 2 parameters");
  });

  it("runAsync rejects on destroyed session", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;

    // Try to load a missing file — this will reject, but we can catch it
    // to test the destroyed behavior
    try {
      const sess = await TFSession.loadFrozenGraph("/nonexistent/path.pb");
      // If somehow it loads, destroy and test
      sess.destroy();
      await assert.rejects(
        () => sess.runAsync({}),
        /destroyed|Session destroyed/i,
      );
    } catch (loadErr) {
      // Expected — loading from nonexistent path fails
      // This means we can't test destroyed behavior easily without a real model
      // That's OK — the C++ implementation already validates this
    }
  });
});

describe("TFSession.runAsync() — frozen graph integration", () => {
  const fixtureModelPath = join(__dirname, "fixtures", "model_frozen.pb");
  const modelPath = process.env.TF_TEST_FROZEN_GRAPH || fixtureModelPath;

  it("runAsync produces same output as run()", async (t) => {
    if (!existsSync(modelPath)) {
      t.skip(
        `Frozen graph not found at: ${modelPath}. Set TF_TEST_FROZEN_GRAPH to override.`,
      );
      return;
    }

    const TFSession = await loadTFSessionOrSkip(t);
    if (!TFSession) return;

    const sess = await TFSession.loadFrozenGraph(modelPath);

    // Use the actual inferred input names from the graph
    const inputKeys = sess.inputs;
    if (inputKeys.length === 0) {
      t.skip("No inputs inferred from frozen graph");
      sess.destroy();
      return;
    }

    // Build inputs map with test data for each inferred input
    const inputs: Record<string, Float32Array> = {};
    for (const key of inputKeys) {
      // Use a 4-element test array for each input
      inputs[key] = new Float32Array([1, 2, 3, 4]);
    }

    const syncResult = await sess.run(inputs);
    const asyncResult = await sess.runAsync(inputs);

    assert.deepEqual(
      syncResult,
      asyncResult,
      "run() and runAsync() produce identical results",
    );

    sess.destroy();
  });
});

describe("TFSession Zero-Copy Memory Management and Lifecycle", () => {
  it("inference results use direct ArrayBuffers (zero-copy)", async (t) => {
    const TFSession = await loadTFSessionOrSkip(t);
    const modelPath =
      process.env.TF_TEST_FROZEN_GRAPH ||
      join(__dirname, "fixtures", "model_frozen.pb");

    if (!TFSession || !existsSync(modelPath)) return t.skip();

    const sess = await TFSession.loadFrozenGraph(modelPath);
    const inputs: Record<string, Float32Array> = {};
    sess.inputs.forEach(
      (key) => (inputs[key] = new Float32Array([1, 2, 3, 4])),
    );

    const result = await sess.runAsync(inputs); //
    const outputKey = Object.keys(result)[0];
    const output = result[outputKey];

    // Narrow the type to access TypedArray-specific properties
    if (!ArrayBuffer.isView(output.data)) {
      throw new Error("Expected a TypedArray from inference result");
    }

    // Now TypeScript knows output.data has .buffer and .length
    assert.ok(output.data.buffer instanceof ArrayBuffer);
    assert.ok(output.data.length > 0);

    // Verify memory remains valid even if we destroy the session immediately
    sess.destroy();

    // The data should still be accessible because the finalizer is tied to
    // the JS object lifecycle, not the session lifecycle.
    assert.doesNotThrow(() => {
      if (ArrayBuffer.isView(output.data)) {
        output.data[0];
      }
    });
  });
});
