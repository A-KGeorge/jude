"use strict";

import nodeGypBuild from "node-gyp-build";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

type NativeSharedTensorCtor = new (maxBytes: number) => any;

let NativeSharedTensor: NativeSharedTensorCtor;
try {
  const addon = nodeGypBuild(join(__dirname, "..")) as any;
  NativeSharedTensor = addon.SharedTensor ?? addon;
} catch (e) {
  try {
    const addon = nodeGypBuild(join(__dirname, "..", "..")) as any;
    NativeSharedTensor = addon.SharedTensor ?? addon;
  } catch (err: any) {
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

const TYPED_ARRAY_CTORS: Record<number, any> = {
  [DType.FLOAT32]: Float32Array,
  [DType.FLOAT64]: Float64Array,
  [DType.INT32]: Int32Array,
  [DType.INT64]: BigInt64Array,
  [DType.UINT8]: Uint8Array,
  [DType.INT8]: Int8Array,
  [DType.UINT16]: Uint16Array,
  [DType.INT16]: Int16Array,
  [DType.BOOL]: Uint8Array,
};

export interface TensorResult {
  shape: number[];
  dtype: DType;
  data: TypedArray;
  version: number;
}

function wrap(result: any): TensorResult | null {
  if (!result) return null;

  const Ctor = TYPED_ARRAY_CTORS[result.dtype];
  if (!Ctor) throw new Error(`Unsupported DType ID: ${result.dtype}`);

  const buf = result.buffer;
  let ab: ArrayBufferLike;
  let byteOffset = 0;

  if (ArrayBuffer.isView(buf)) {
    ab = buf.buffer;
    byteOffset = buf.byteOffset;
  } else {
    ab = buf;
  }

  return {
    shape: result.shape,
    dtype: result.dtype,
    version: result.version,
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

  get isPinned(): boolean {
    return Boolean(this._native.isPinned);
  }

  /**
   * Write a tensor from an existing JS buffer.
   * Subject to V8's ~2 GB TypedArray ceiling — use fill() for large tensors.
   */
  write(shape: number[], dtype: DType, buffer: ArrayBuffer | TypedArray): void {
    this._native.write(shape, dtype, buffer);
  }

  /**
   * Fill every element of the tensor with a scalar value, entirely in C++.
   *
   * No V8 buffer is allocated — the data materialises directly in the mmap
   * region. This is the correct path for tensors larger than ~2 GB where
   * Buffer.allocUnsafe() would throw RangeError.
   *
   * For INT64 the value is cast from a JS number (double). Values outside
   * the safe integer range (±2^53) will lose precision — BigInt support
   * can be added if that becomes a requirement.
   *
   * @example
   * // 10 GB float32 tensor — no V8 buffer needed
   * const GB = 1024 * 1024 * 1024;
   * const seg = new SharedTensorSegment(10 * GB);
   * seg.fill([10 * GB / 4], DType.FLOAT32, 0.0);
   */
  fill(shape: number[], dtype: DType, value: number): void {
    this._native.fill(shape, dtype, value);
  }

  /**
   * Zero-copy read. Returns a view directly into shared memory.
   * Valid until the next write() or fill() or destroy().
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
   * Native code spins briefly then parks until a writer commit wakes it.
   */
  async readWait(): Promise<TensorResult | null> {
    return wrap(await this._native.readWait());
  }

  /**
   * Promise-based copy read with spin-then-park behaviour.
   */
  async readCopyWait(): Promise<TensorResult | null> {
    return wrap(await this._native.readCopyWait());
  }

  /**
   * Page-lock the mapped region for CUDA H2D zero-copy paths.
   * Returns true on success, false if the OS denies the lock (non-fatal).
   */
  pin(): boolean {
    return Boolean(this._native.pin());
  }

  unpin(): void {
    this._native.unpin();
  }

  destroy(): void {
    this._native.destroy();
  }
}
