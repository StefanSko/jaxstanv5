// One web worker per chain: the library is a pure function, so all
// parallelism lives out here.

import { loadJstan } from "./glue.js";

self.onmessage = async (event) => {
  const { wasmUrl, model, data, settings, seed, chainId } = event.data;
  try {
    const jstan = await loadJstan(wasmUrl);
    const response = jstan.run({
      command: "sample",
      model,
      data,
      settings,
      seed,
      chain_id: chainId,
    });
    if (response.startsWith('{"error"')) {
      self.postMessage({ chainId, error: JSON.parse(response) });
      return;
    }
    const lines = response.split("\n");
    const header = JSON.parse(lines[0]);
    const draws = lines.slice(1, -1).map((line) => JSON.parse(line).values);
    const trailer = JSON.parse(lines[lines.length - 1]).trailer;
    self.postMessage({ chainId, header, draws, trailer });
  } catch (error) {
    self.postMessage({ chainId, error: { error: "WorkerFailure", message: String(error) } });
  }
};
