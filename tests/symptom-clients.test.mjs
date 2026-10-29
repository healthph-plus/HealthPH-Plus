import assert from "node:assert/strict";
import test from "node:test";

import {
  createDiseaseClient, DiseaseClientError,
} from "../disease-detection-service/examples/disease-client.mjs";
import {
  createXlmrDiseaseClient, XlmrDiseaseClientError,
} from "../xlmr-disease-detection-service/examples/xlmr-disease-client.mjs";

const factories = [
  ["ELECTRA", createDiseaseClient, DiseaseClientError],
  ["XLM-R", createXlmrDiseaseClient, XlmrDiseaseClientError],
];
const text = "😷 walang ubo; ubo";
const mention = {
  text: "ubo", normalized_symptom: "cough", start_index: 9, end_index: 12,
  confidence: 0.93, negated: true,
};

function prediction(symptoms = []) {
  return {
    labels: ["AURI"],
    probabilities: { AURI: 0.9, PN: 0.1, TB: 0.2, COVID: 0.3 },
    thresholds_used: { AURI: 0.5, PN: 0.5, TB: 0.5, COVID: 0.5 },
    model_version: "disease-version",
    symptoms,
    symptom_model_version: "symptom-version",
  };
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status, headers: { "Content-Type": "application/json" },
  });
}

for (const [name, factory, ErrorType] of factories) {
  const client = (settings = {}) => factory({ serviceUrl: "http://in-memory.invalid", ...settings });

  test(`${name}: exact Unicode text, negation and both versions survive the request`, async (t) => {
    const combined = prediction([mention]);
    const calls = [];
    t.mock.method(globalThis, "fetch", async (url, options) => {
      calls.push({ url: url.href, ...options });
      return jsonResponse(combined);
    });
    const result = await client().predict(text);
    assert.deepEqual(result, combined);
    assert.equal(Array.from(text).slice(mention.start_index, mention.end_index).join(""), mention.text);
    assert.notEqual(text.slice(mention.start_index, mention.end_index), mention.text);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "http://in-memory.invalid/v1/disease/predict");
    assert.equal(calls[0].method, "POST");
    assert.deepEqual(calls[0].headers, { "Content-Type": "application/json" });
    assert.deepEqual(JSON.parse(calls[0].body), { text });
    assert.ok(calls[0].signal instanceof AbortSignal);
  });

  test(`${name}: null normalization and an empty symptom list are valid`, async (t) => {
    const combined = prediction([{ ...mention, normalized_symptom: null, negated: false }]);
    t.mock.method(globalThis, "fetch", async () => jsonResponse(combined));
    assert.deepEqual(await client().predict(text), combined);
    globalThis.fetch.mock.mockImplementation(async () => jsonResponse(prediction()));
    assert.deepEqual(await client().predict("feeling okay"), prediction());
  });

  test(`${name}: batches retain input order, duplicates and repeated mentions`, async (t) => {
    const texts = [text, "feeling okay", text];
    const repeated = prediction([
      mention,
      { ...mention, start_index: 14, end_index: 17, negated: false },
    ]);
    const response = { results: [repeated, prediction(), repeated] };
    t.mock.method(globalThis, "fetch", async (url, options) => {
      assert.equal(url.pathname, "/v1/disease/predict-batch");
      assert.deepEqual(JSON.parse(options.body), { texts });
      return jsonResponse(response);
    });
    assert.deepEqual(await client().predictBatch(texts), response);
  });

  test(`${name}: batch symptoms are validated against each input position`, async (t) => {
    t.mock.method(globalThis, "fetch", async () => jsonResponse({
      results: [prediction(), prediction([mention])],
    }));
    await assert.rejects(client().predictBatch([text, "feeling okay"]), { code: "INVALID_RESPONSE" });
    globalThis.fetch.mock.mockImplementation(async () => jsonResponse({ results: [] }));
    await assert.rejects(client().predictBatch([text]), { code: "INVALID_RESPONSE" });
  });

  const invalidPredictions = [
    ["missing symptoms", (result) => { delete result.symptoms; }],
    ["null symptoms", (result) => { result.symptoms = null; }],
    ["missing symptom version", (result) => { delete result.symptom_model_version; }],
    ["empty symptom version", (result) => { result.symptom_model_version = ""; }],
    ["nonstring symptom version", (result) => { result.symptom_model_version = 1; }],
    ["missing disease version", (result) => { delete result.model_version; }],
    ["empty disease version", (result) => { result.model_version = ""; }],
    ["nonstring disease version", (result) => { result.model_version = 1; }],
    ...Object.keys(mention).map((key) => [`missing symptom ${key}`, (result) => {
      delete result.symptoms[0][key];
    }]),
    ["extra symptom field", (result) => { result.symptoms[0].extra = true; }],
    ["wrong source text", (result) => { result.symptoms[0].text = "cough"; }],
    ["empty source text", (result) => { result.symptoms[0].text = ""; }],
    ["string negation", (result) => { result.symptoms[0].negated = "true"; }],
    ["empty normalization", (result) => { result.symptoms[0].normalized_symptom = ""; }],
    ["nonstring normalization", (result) => { result.symptoms[0].normalized_symptom = 7; }],
    ["negative confidence", (result) => { result.symptoms[0].confidence = -0.1; }],
    ["excessive confidence", (result) => { result.symptoms[0].confidence = 1.1; }],
    ["null confidence", (result) => { result.symptoms[0].confidence = null; }],
    ["string confidence", (result) => { result.symptoms[0].confidence = "0.9"; }],
    ["fractional offset", (result) => { result.symptoms[0].start_index = 9.1; }],
    ["negative offset", (result) => { result.symptoms[0].start_index = -1; }],
    ["string offset", (result) => { result.symptoms[0].end_index = "12"; }],
    ["unsafe offset", (result) => { result.symptoms[0].end_index = Number.MAX_SAFE_INTEGER + 1; }],
    ["empty range", (result) => { result.symptoms[0].end_index = 9; }],
    ["reversed range", (result) => { result.symptoms[0].end_index = 8; }],
    ["out of bounds", (result) => { result.symptoms[0].end_index = 100; }],
    ["UTF-16 offset", (result) => { result.symptoms[0].start_index = 10; result.symptoms[0].end_index = 13; }],
    ["overlapping ranges", (result) => { result.symptoms.push({ ...mention }); }],
    ["unordered ranges", (result) => {
      result.symptoms.unshift({ ...mention, start_index: 14, end_index: 17 });
    }],
    ["null mention", (result) => { result.symptoms = [null]; }],
  ];
  for (const [description, change] of invalidPredictions) {
    test(`${name}: rejects ${description}`, async (t) => {
      const result = prediction([{ ...mention }]);
      change(result);
      t.mock.method(globalThis, "fetch", async () => jsonResponse(result));
      await assert.rejects(client().predict(text), (error) => {
        assert.ok(error instanceof ErrorType);
        assert.equal(error.code, "INVALID_RESPONSE");
        assert.equal(error.status, 200);
        return true;
      });
    });
  }

  for (const status of [422, 500, 503]) {
    test(`${name}: HTTP ${status} exposes server detail without retries`, async (t) => {
      let calls = 0;
      const details = { detail: "symptom inference failed" };
      t.mock.method(globalThis, "fetch", async () => {
        calls += 1;
        return jsonResponse(details, status);
      });
      await assert.rejects(client().predict(text), (error) => {
        assert.ok(error instanceof ErrorType);
        assert.equal(error.code, "HTTP_ERROR");
        assert.equal(error.status, status);
        assert.deepEqual(error.details, details);
        return true;
      });
      assert.equal(calls, 1);
    });
  }

  test(`${name}: invalid JSON and non-JSON HTTP errors remain distinct`, async (t) => {
    t.mock.method(globalThis, "fetch", async () => new Response("not JSON"));
    await assert.rejects(client().predict(text), { code: "INVALID_RESPONSE", status: 200 });
    globalThis.fetch.mock.mockImplementation(async () => new Response("unavailable", { status: 503 }));
    await assert.rejects(client().predict(text), {
      code: "HTTP_ERROR", status: 503, details: "unavailable",
    });
  });

  test(`${name}: connection failures remain errors`, async (t) => {
    t.mock.method(globalThis, "fetch", async () => { throw new TypeError("failed fetch"); });
    await assert.rejects(client().predict(text), { code: "CONNECTION_ERROR" });
  });

  test(`${name}: timeout aborts pending fetch without any network calls`, async (t) => {
    let aborted = false;
    t.mock.method(globalThis, "fetch", (url, { signal }) => new Promise((resolve, reject) => {
      signal.addEventListener("abort", () => {
        aborted = true;
        reject(new DOMException("The operation was aborted", "AbortError"));
      }, { once: true });
    }));
    await assert.rejects(client({ timeoutMs: 10 }).predict(text), { code: "TIMEOUT" });
    assert.equal(aborted, true);
  });

  test(`${name}: timeout also covers a stalled response body`, async (t) => {
    t.mock.method(globalThis, "fetch", async (url, { signal }) => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"labels":'));
        signal.addEventListener("abort", () => {
          controller.error(new DOMException("The operation was aborted", "AbortError"));
        }, { once: true });
      },
    })));
    await assert.rejects(client({ timeoutMs: 10 }).predict(text), { code: "TIMEOUT" });
  });
}
