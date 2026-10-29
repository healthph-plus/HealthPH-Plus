import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { createXlmrDiseaseClient, XlmrDiseaseClientError } from "../examples/xlmr-disease-client.mjs";

const prediction = {
  labels: ["AURI"],
  probabilities: { AURI: 0.9, PN: 0.1, TB: 0.2, COVID: 0.3 },
  thresholds_used: { AURI: 0.5, PN: 0.5, TB: 0.5, COVID: 0.5 },
  model_version: "test-version",
  symptoms: [],
  symptom_model_version: "test-symptom-version",
};

async function serve(t, handler) {
  const server = createServer(handler);
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  return { server, serviceUrl: `http://127.0.0.1:${server.address().port}` };
}

test("single and batch requests preserve the JSON contract", async (t) => {
  const requests = [];
  const { serviceUrl } = await serve(t, async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    const data = JSON.parse(body);
    requests.push({ path: request.url, data, type: request.headers["content-type"] });
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify(request.url.endsWith("predict-batch")
      ? { results: data.texts.map(() => prediction) } : prediction));
  });
  const client = createXlmrDiseaseClient({ serviceUrl });
  assert.deepEqual(await client.predict("  ubo 😷  "), prediction);
  assert.deepEqual(await client.predictBatch(["a", "b"]), { results: [prediction, prediction] });
  assert.deepEqual(requests, [
    { path: "/v1/disease/predict", data: { text: "  ubo 😷  " }, type: "application/json" },
    { path: "/v1/disease/predict-batch", data: { texts: ["a", "b"] }, type: "application/json" },
  ]);
});

for (const status of [422, 500, 503]) {
  test(`HTTP ${status} is exposed without retries`, async (t) => {
    let calls = 0;
    const { serviceUrl } = await serve(t, (request, response) => {
      calls += 1;
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ detail: "service error" }));
    });
    await assert.rejects(createXlmrDiseaseClient({ serviceUrl }).predict("ubo"), (error) => {
      assert.ok(error instanceof XlmrDiseaseClientError);
      assert.equal(error.code, "HTTP_ERROR");
      assert.equal(error.status, status);
      assert.deepEqual(error.details, { detail: "service error" });
      return true;
    });
    assert.equal(calls, 1);
  });
}

for (const body of [
  "not JSON", "{}", "null", "[]",
  JSON.stringify({ ...prediction, probabilities: { AURI: 2 } }),
  JSON.stringify({ ...prediction, labels: [] }),
  JSON.stringify({ ...prediction, labels: ["AURI", "AURI"] }),
  JSON.stringify({ ...prediction, thresholds_used: { ...prediction.thresholds_used, EXTRA: 0.5 } }),
  JSON.stringify({ ...prediction, model_version: "" }),
]) {
  test(`reject malformed success: ${body}`, async (t) => {
    const { serviceUrl } = await serve(t, (request, response) => response.end(body));
    await assert.rejects(createXlmrDiseaseClient({ serviceUrl }).predict("ubo"), { code: "INVALID_RESPONSE" });
  });
}

test("reject an incomplete batch response", async (t) => {
  const { serviceUrl } = await serve(t, (request, response) => response.end('{"results":[]}'));
  await assert.rejects(createXlmrDiseaseClient({ serviceUrl }).predictBatch(["ubo"]), { code: "INVALID_RESPONSE" });
});

test("timeout covers a stalled response body", async (t) => {
  const { serviceUrl } = await serve(t, (request, response) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.write('{"labels":');
  });
  await assert.rejects(createXlmrDiseaseClient({ serviceUrl, timeoutMs: 50 }).predict("ubo"), { code: "TIMEOUT" });
});

test("connection failures have a distinct error code", async (t) => {
  const { server, serviceUrl } = await serve(t, () => {});
  await new Promise((resolve) => server.close(resolve));
  await assert.rejects(createXlmrDiseaseClient({ serviceUrl }).predict("ubo"), { code: "CONNECTION_ERROR" });
});

test("inclusive threshold equality and empty labels are valid", async (t) => {
  const boundary = { ...prediction, probabilities: { ...prediction.probabilities, AURI: 0.5 } };
  const empty = { ...prediction, labels: [], probabilities: { ...prediction.probabilities, AURI: 0.1 } };
  const { serviceUrl } = await serve(t, (request, response) => {
    response.end(JSON.stringify(request.url.endsWith("predict-batch")
      ? { results: [boundary, empty] } : boundary));
  });
  const client = createXlmrDiseaseClient({ serviceUrl });
  assert.deepEqual(await client.predict("ubo"), boundary);
  assert.deepEqual(await client.predictBatch(["ubo", "okay"]), { results: [boundary, empty] });
});

test("environment URL is used and explicit URL takes precedence", async (t) => {
  const previous = process.env.HEALTHPH_XLMR_DISEASE_URL;
  t.after(() => {
    if (previous === undefined) delete process.env.HEALTHPH_XLMR_DISEASE_URL;
    else process.env.HEALTHPH_XLMR_DISEASE_URL = previous;
  });
  const { serviceUrl } = await serve(t, (request, response) => response.end(JSON.stringify(prediction)));
  process.env.HEALTHPH_XLMR_DISEASE_URL = serviceUrl;
  assert.deepEqual(await createXlmrDiseaseClient().predict("ubo"), prediction);
  process.env.HEALTHPH_XLMR_DISEASE_URL = "invalid URL";
  assert.deepEqual(await createXlmrDiseaseClient({ serviceUrl }).predict("ubo"), prediction);
});

test("invalid URL and timeout settings fail immediately", () => {
  for (const serviceUrl of ["file:///tmp/model", "not a URL"]) {
    assert.throws(() => createXlmrDiseaseClient({ serviceUrl }), TypeError);
  }
  for (const timeoutMs of [0, -1, 1.5, Infinity, 2_147_483_648, "30000"]) {
    assert.throws(() => createXlmrDiseaseClient({ timeoutMs }), TypeError);
  }
});

test("real service single, batch, and validation", {
  skip: process.env.HEALTHPH_TEST_REAL_SERVICE !== "1",
}, async () => {
  const client = createXlmrDiseaseClient();
  const text = "ubo at lagnat";
  const single = await client.predict(text);
  const { results } = await client.predictBatch([text, "feeling okay", text]);
  assert.match(single.model_version, /^sha256:[0-9a-f]{64}$/);
  assert.equal(results.length, 3);
  for (const index of [0, 2]) {
    assert.deepEqual(results[index].labels, single.labels);
    assert.equal(results[index].model_version, single.model_version);
    for (const label of Object.keys(single.probabilities)) {
      assert.ok(Math.abs(results[index].probabilities[label] - single.probabilities[label]) <= 2e-5);
    }
  }
  await assert.rejects(client.predict(" "), { code: "HTTP_ERROR", status: 422 });
  const { stdout } = await promisify(execFile)(process.execPath, [
    fileURLToPath(new URL("../examples/xlmr-disease-client.mjs", import.meta.url)), text,
  ], { env: process.env, timeout: 35_000 });
  const cliResult = JSON.parse(stdout);
  assert.deepEqual(cliResult.labels, single.labels);
  assert.equal(cliResult.model_version, single.model_version);
});


const symptom = {
  text: "ubo", normalized_symptom: "cough", start_index: 9, end_index: 12,
  confidence: 0.93, negated: true,
};

test("symptom offsets count code points and preserve negation", async (t) => {
  const text = "😷 walang ubo";
  const combined = { ...prediction, symptoms: [symptom] };
  const { serviceUrl } = await serve(t, (request, response) => {
    response.end(JSON.stringify(request.url.endsWith("predict-batch")
      ? { results: [combined, prediction] } : combined));
  });
  const client = createXlmrDiseaseClient({ serviceUrl });
  assert.deepEqual(await client.predict(text), combined);
  assert.equal(Array.from(text).slice(symptom.start_index, symptom.end_index).join(""), "ubo");
  assert.deepEqual(await client.predictBatch([text, "okay"]), { results: [combined, prediction] });
});

for (const change of [
  { symptoms: undefined }, { symptoms: null }, { symptom_model_version: "" },
  { symptoms: [{ ...symptom, start_index: 10 }] },
  { symptoms: [{ ...symptom, end_index: 13 }] },
  { symptoms: [{ ...symptom, normalized_symptom: 1 }] },
  { symptoms: [{ ...symptom, negated: "true" }] },
  { symptoms: [{ ...symptom, confidence: -0.1 }] },
  { symptoms: [{ ...symptom, start_index: 9.1 }] },
  { symptoms: [{ ...symptom, text: "cough" }] },
  { symptoms: [{ ...symptom, extra: true }] },
  { symptoms: [symptom, symptom] },
]) {
  test(`reject invalid symptom response ${JSON.stringify(change)}`, async (t) => {
    const { serviceUrl } = await serve(t, (request, response) => {
      response.end(JSON.stringify({ ...prediction, ...change }));
    });
    await assert.rejects(createXlmrDiseaseClient({ serviceUrl }).predict("😷 walang ubo"), { code: "INVALID_RESPONSE" });
  });
}

test("unmapped symptom phrases retain null normalization", async (t) => {
  const combined = { ...prediction, symptoms: [{ ...symptom, normalized_symptom: null }] };
  const { serviceUrl } = await serve(t, (request, response) => response.end(JSON.stringify(combined)));
  assert.deepEqual(await createXlmrDiseaseClient({ serviceUrl }).predict("😷 walang ubo"), combined);
});
