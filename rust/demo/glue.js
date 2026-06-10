// Zero-dependency JS glue for the jaxstanv5-core wasm ABI.
//
// Memory contract (see rust/jaxstanv5-core/src/wasm_abi.rs):
// request bytes in via jstan_alloc, response out of jstan_run with the
// length written to a 4-byte slot, all buffers released via jstan_dealloc.

export async function loadJstan(wasmUrl) {
  const response = await fetch(wasmUrl);
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(response, {}));
  } catch {
    // Server did not send application/wasm; fall back to ArrayBuffer.
    const bytes = await (await fetch(wasmUrl)).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, {}));
  }
  const { memory, jstan_alloc, jstan_dealloc, jstan_run } = instance.exports;

  function run(requestObject) {
    const requestBytes = new TextEncoder().encode(JSON.stringify(requestObject));
    const requestPtr = jstan_alloc(requestBytes.length);
    new Uint8Array(memory.buffer, requestPtr, requestBytes.length).set(requestBytes);
    const lengthPtr = jstan_alloc(4);
    const responsePtr = jstan_run(requestPtr, requestBytes.length, lengthPtr);
    // memory.buffer may have been detached by growth during the call;
    // re-create views afterwards.
    const responseLength = new DataView(memory.buffer).getUint32(lengthPtr, true);
    const responseBytes = new Uint8Array(memory.buffer, responsePtr, responseLength).slice();
    jstan_dealloc(requestPtr, requestBytes.length);
    jstan_dealloc(lengthPtr, 4);
    jstan_dealloc(responsePtr, responseLength);
    return new TextDecoder().decode(responseBytes);
  }

  return { run };
}
