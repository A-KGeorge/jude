# jude-map

High-performance shared tensor memory for Node.js using a native N-API addon.

`jude-map` provides a memory-mapped tensor segment with seqlock-based consistency, zero-copy reads, copy reads, and async wait-based reads. It also includes optional page pinning helpers for CUDA host-to-device workflows.

## Features

- Shared memory segment for tensor payloads
- Fast writes with seqlock commit semantics
- `read()` for zero-copy reads (ArrayBuffer-backed)
- `readCopy()` for safe copied reads
- `readWait()` / `readCopyWait()` for async spin-then-park reader behavior
- Optional `pin()` / `unpin()` for page-locked memory usage
- Typed TS API with `DType` enum mapping

## Requirements

- Node.js >= 18
- npm >= 11.5.1
- Native build toolchain for `node-gyp` (C++ compiler + Python)

## Installation

```bash
npm install jude-map
```

If building from source in this repository:

```bash
npm install
npm run build
```

## Quick Start

```ts
import { SharedTensorSegment, DType } from "jude-map";

const seg = new SharedTensorSegment(4 * 1024 * 1024); // 4 MB

// Write [2,3] float32 tensor
const input = new Float32Array([1, 2, 3, 4, 5, 6]);
seg.write([2, 3], DType.FLOAT32, input);

// Zero-copy read
const r0 = seg.read();
if (r0) {
  console.log(r0.shape, r0.dtype, r0.version);
  console.log((r0.data as Float32Array)[5]);
}

// Copy read
const r1 = seg.readCopy();
if (r1) {
  console.log(r1.shape, r1.data.length);
}

seg.destroy();
```

## API

### `new SharedTensorSegment(maxBytes: number)`

Creates a mapped tensor segment with capacity `maxBytes` (plus internal metadata header).

### Properties

- `byteCapacity: number` - writable tensor byte capacity
- `isPinned: boolean` - whether the mapping is currently page-locked

### Methods

- `write(shape: number[], dtype: DType, buffer: ArrayBuffer | TypedArray): void`
- `read(): TensorResult | null`
- `readCopy(): TensorResult | null`
- `readWait(): Promise<TensorResult | null>`
- `readCopyWait(): Promise<TensorResult | null>`
- `pin(): boolean`
- `unpin(): void`
- `destroy(): void`

### `DType`

```ts
enum DType {
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
```

### `TensorResult`

```ts
interface TensorResult {
  shape: number[];
  dtype: DType;
  data: TypedArray;
  version: number;
}
```

## CUDA / Page Pinning Notes

`pin()` attempts to page-lock the mapped region so GPU runtimes can DMA directly from host memory without an extra staging copy. This may require elevated privileges depending on OS configuration. If pinning is unavailable, `pin()` returns `false`.

Always call `unpin()` when done with long-lived pinned buffers.

## Development

Build TypeScript only:

```bash
npm run build:ts
```

Build native addon only:

```bash
npm run build:native
```

Build everything:

```bash
npm run build
```

Run tests:

```bash
npm run test
```

Run install-path smoke check:

```bash
npm run test:install
```

## License

Apache-2.0
