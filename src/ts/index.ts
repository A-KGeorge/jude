"use strict";

import nodeGypBuild from "node-gyp-build";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// Import the native addon - update the path as necessary for your project
// const {
//   SharedTensor: NativeSharedTensor,
// } = require("./build/Release/tensor_bridge");

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

type NativeSharedTensorCtor = new (maxBytes: number) => any;

let NativeSharedTensor: NativeSharedTensorCtor;
// Load the addon using node-gyp-build
try {
  // First, try the path that works when installed
  const addon = nodeGypBuild(join(__dirname, "..")) as any;
  NativeSharedTensor = addon.SharedTensor ?? addon;
} catch (e) {
  try {
    // If that fails, try the path that works locally during testing/dev
    const addon = nodeGypBuild(join(__dirname, "..", "..")) as any;
    NativeSharedTensor = addon.SharedTensor ?? addon;
  } catch (err: any) {
    // If both fail, throw a more informative error
    console.error("Failed to load native SharedTensor module.");
    console.error("Tried using both relative paths.");
    console.error(
      "Attempt 1 error (installed path ../):",
      (e as Error).message,
    );
    console.error("Attempt 2 error (local path ../../):", err.message);
    throw new Error(
      `Could not load native module. Is the build complete? Search paths tried: ${join(
        __dirname,
        "..",
      )} and ${join(__dirname, "..", "..")}`,
    );
  }
}

if (typeof NativeSharedTensor !== "function") {
  throw new Error(
    "Native module loaded, but SharedTensor constructor was not found.",
  );
}

/**
 * Numeric DType enum to match the C++ DType enum class exactly.
 * Passing integers over the N-API bridge is significantly faster than strings.
 */
export enum DType {
  FLOAT32 = 0,
  FLOAT64 = 1,
  INT32 = 2,
  INT64 = 3,
  UINT8 = 4,
  INT8 = 5,
  UINT16 = 6,
  INT16 = 7,
  BOOL = 8,
}

type TypedArray =
  | Float32Array
  | Float64Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array;

/**
 * Mapping of DType to TypedArray constructors.
 * Using the numeric enum as a key is faster than string lookups.
 */
const TYPED_ARRAY_CTORS: Record<number, any> = {
  [DType.FLOAT32]: Float32Array,
  [DType.FLOAT64]: Float64Array,
  [DType.INT32]: Int32Array,
  [DType.INT64]: BigInt64Array,
  [DType.UINT8]: Uint8Array,
  [DType.INT8]: Int8Array,
  [DType.UINT16]: Uint16Array,
  [DType.INT16]: Int16Array,
  [DType.BOOL]: Uint8Array, // Packed as uint8
};

export interface TensorResult {
  shape: number[];
  dtype: DType;
  data: TypedArray;
  version: number; // Added to support the safety check we implemented in C++
}

/**
 * Internal helper to wrap raw N-API results into TypedArrays.
 */
function wrap(result: any): TensorResult | null {
  if (!result) return null;

  const Ctor = TYPED_ARRAY_CTORS[result.dtype];
  if (!Ctor) throw new Error(`Unsupported DType ID: ${result.dtype}`);

  const buf = result.buffer;

  // 2. Change type to ArrayBufferLike to accept both buffer types
  let ab: ArrayBufferLike;
  let byteOffset = 0;

  if (ArrayBuffer.isView(buf)) {
    ab = buf.buffer; // This now works with ArrayBufferLike
    byteOffset = buf.byteOffset;
  } else {
    ab = buf;
  }

  return {
    shape: result.shape,
    dtype: result.dtype,
    version: result.version,
    // The TypedArray constructor natively accepts ArrayBufferLike
    data: new Ctor(
      ab,
      byteOffset,
      result.buffer.byteLength / Ctor.BYTES_PER_ELEMENT,
    ),
  };
}

export class SharedTensorSegment {
  private _native: any;

  constructor(maxBytes: number) {
    this._native = new NativeSharedTensor(maxBytes);
  }

  get byteCapacity(): number {
    return this._native.byteCapacity;
  }

  /**
   * Whether the backing mmap region is currently page-locked for GPU DMA.
   */
  get isPinned(): boolean {
    return Boolean(this._native.isPinned);
  }

  /**
   * Write a tensor. Use the DType enum for maximum performance.
   */
  write(shape: number[], dtype: DType, buffer: ArrayBuffer | TypedArray): void {
    this._native.write(shape, dtype, buffer);
  }

  /**
   * Zero-copy read. Returns a view directly into shared memory.
   * Check 'version' to ensure data wasn't torn during processing.
   */
  read(): TensorResult | null {
    return wrap(this._native.read());
  }

  /**
   * Safe read. Copies data into a fresh buffer.
   */
  readCopy(): TensorResult | null {
    return wrap(this._native.readCopy());
  }

  /**
   * Promise-based zero-copy read.
   * Native code spins briefly and then parks until a writer commit wakes it.
   */
  async readWait(): Promise<TensorResult | null> {
    return wrap(await this._native.readWait());
  }

  /**
   * Promise-based copy read with spin-then-park behavior.
   */
  async readCopyWait(): Promise<TensorResult | null> {
    return wrap(await this._native.readCopyWait());
  }

  /**
   * Attempt to page-lock the mapped region for CUDA H2D zero-copy paths.
   * Returns true when the mapping is pinned, false when pinning is not available.
   */
  pin(): boolean {
    return Boolean(this._native.pin());
  }

  /**
   * Release page lock if previously pinned.
   */
  unpin(): void {
    this._native.unpin();
  }

  destroy(): void {
    this._native.destroy();
  }
}
