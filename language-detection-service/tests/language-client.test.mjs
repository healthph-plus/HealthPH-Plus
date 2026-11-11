import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { createLanguageClient, LanguageClientError } from "../examples/language-client.mjs";

const prediction = {
  language: "hil",
  cleaned_text: "ubo",
  is_supported: true,
  prediction_source: "keyword_override",
  fasttext_language: "tl",
  fasttext_confidence: 0.9,
  fasttext_confidence_percentage: 90,
};

const twoLanguagePrediction = {
  ...prediction,
  fasttext_language: "tl",
  fasttext_confidence: 0.6,
  fasttext_confidence_percentage: 60,
  fasttext_predictions: [
    { language: "tl", confidence: 0.6, confidence_percentage: 60 },
    { language: "en", confidence: 0.3, confidence_percentage: 30 },
  ],
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
  const client = createLanguageClient({ serviceUrl });
  assert.deepEqual(await client.predict("  ubo 😷  "), prediction);
  assert.deepEqual(await client.predictBatch(["a", "b"]), { results: [prediction, prediction] });
  assert.deepEqual(requests, [
    { path: "/v1/language/predict", data: { text: "  ubo 😷  " }, type: "application/json" },
    { path: "/v1/language/predict-batch", data: { texts: ["a", "b"] }, type: "application/json" },
  ]);
});

test("optional candidates forward only specified options for single and batch requests", async (t) => {
  const requests = [];
  const { serviceUrl } = await serve(t, async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    const data = JSON.parse(body);
    requests.push(data);
    const result = data.top_k === 2 ? {
      ...twoLanguagePrediction,
      fasttext_predictions: twoLanguagePrediction.fasttext_predictions.filter((candidate, index) =>
        index === 0 || candidate.confidence >= (data.secondary_min_confidence ?? 0.3)),
    } : prediction;
    response.end(JSON.stringify(data.texts ? { results: data.texts.map(() => result) } : result));
  });
  const client = createLanguageClient({ serviceUrl });
  assert.deepEqual(await client.predict("mixed", { topK: 2 }), twoLanguagePrediction);
  const filtered = {
    ...twoLanguagePrediction, fasttext_predictions: twoLanguagePrediction.fasttext_predictions.slice(0, 1),
  };
  assert.deepEqual(await client.predictBatch(["mixed", "mixed"], {
    topK: 2, secondaryMinConfidence: 0.4,
  }), { results: [filtered, filtered] });
  assert.deepEqual(await client.predict("one", { secondaryMinConfidence: 0 }), prediction);
  assert.deepEqual(requests, [
    { text: "mixed", top_k: 2 },
    { texts: ["mixed", "mixed"], top_k: 2, secondary_min_confidence: 0.4 },
    { text: "one", secondary_min_confidence: 0 },
  ]);
});

test("candidate validation allows a top score below the secondary threshold", async (t) => {
  const lowConfidence = {
    ...twoLanguagePrediction, fasttext_confidence: 0.2, fasttext_confidence_percentage: 20,
    fasttext_predictions: [{ language: "tl", confidence: 0.2, confidence_percentage: 20 }],
  };
  const { serviceUrl } = await serve(t, (_request, response) => response.end(JSON.stringify(lowConfidence)));
  assert.deepEqual(await createLanguageClient({ serviceUrl }).predict("mixed", { topK: 2 }), lowConfidence);
});

test("candidate validation honors a custom lower threshold", async (t) => {
  const lowSecondary = {
    ...twoLanguagePrediction,
    fasttext_predictions: [
      { language: "tl", confidence: 0.6, confidence_percentage: 60 },
      { language: "en", confidence: 0.1, confidence_percentage: 10 },
    ],
  };
  const { serviceUrl } = await serve(t, (_request, response) => response.end(JSON.stringify(lowSecondary)));
  assert.deepEqual(await createLanguageClient({ serviceUrl }).predict("mixed", {
    topK: 2, secondaryMinConfidence: 0.1,
  }), lowSecondary);
});

for (const [name, candidates] of [
  ["missing", undefined], ["null", null], ["empty", []], ["not an array", {}],
  ["too many", [...twoLanguagePrediction.fasttext_predictions, { language: "fr", confidence: 0.3 }]],
  ["missing score", [{ language: "tl" }]],
  ["string score", [{ language: "tl", confidence: "0.6" }]],
  ["non-finite score encoded as null", [{ language: "tl", confidence: null }]],
  ["score out of range", [{ language: "tl", confidence: 1.1 }]],
  ["negative score", [{ language: "tl", confidence: -0.1 }]],
  ["percentage out of range", [{ language: "tl", confidence: 0.6, confidence_percentage: 101 }]],
  ["negative percentage", [{ language: "tl", confidence: 0.6, confidence_percentage: -1 }]],
  ["string percentage", [{ language: "tl", confidence: 0.6, confidence_percentage: "60" }]],
  ["null percentage", [{ language: "tl", confidence: 0.6, confidence_percentage: null }]],
  ["inconsistent percentage", [{ language: "tl", confidence: 0.6, confidence_percentage: 61 }]],
  ["empty language", [{ language: "", confidence: 0.6 }]],
  ["inconsistent top language", [{ language: "en", confidence: 0.6 }]],
  ["inconsistent top score", [{ language: "tl", confidence: 0.5 }]],
  ["below threshold", [{ language: "tl", confidence: 0.6 }, { language: "en", confidence: 0.2999 }]],
  ["duplicate languages", [{ language: "tl", confidence: 0.6 }, { language: "tl", confidence: 0.3 }]],
  ["unsorted scores", [{ language: "tl", confidence: 0.6 }, { language: "en", confidence: 0.7 }]],
]) {
  test(`reject malformed candidate response: ${name}`, async (t) => {
    const invalid = { ...twoLanguagePrediction, fasttext_predictions: candidates };
    const { serviceUrl } = await serve(t, (_request, response) => response.end(JSON.stringify(invalid)));
    await assert.rejects(createLanguageClient({ serviceUrl }).predict("mixed", { topK: 2 }), {
      code: "INVALID_RESPONSE",
    });
  });
}

test("rounded percentages are preserved for single and batch predictions", async (t) => {
  const rounded = {
    ...twoLanguagePrediction, fasttext_confidence: 0.654321, fasttext_confidence_percentage: 65.43,
    fasttext_predictions: [
      { language: "tl", confidence: 0.654321, confidence_percentage: 65.43 },
      { language: "en", confidence: 0.312345, confidence_percentage: 31.23 },
    ],
  };
  const { serviceUrl } = await serve(t, (request, response) => response.end(JSON.stringify(
    request.url.endsWith("predict-batch") ? { results: [rounded] } : rounded,
  )));
  const client = createLanguageClient({ serviceUrl });
  assert.deepEqual(await client.predict("mixed", { topK: 2 }), rounded);
  assert.deepEqual(await client.predictBatch(["mixed"], { topK: 2 }), { results: [rounded] });
});

test("older service responses without percentage fields remain supported", async (t) => {
  const { fasttext_confidence_percentage, ...legacy } = twoLanguagePrediction;
  legacy.fasttext_predictions = legacy.fasttext_predictions.map(({ language, confidence }) => ({ language, confidence }));
  const { serviceUrl } = await serve(t, (_request, response) => response.end(JSON.stringify(legacy)));
  assert.deepEqual(await createLanguageClient({ serviceUrl }).predict("mixed", { topK: 2 }), legacy);
});

for (const percentage of [-1, 101, "60", null, 61]) {
  test(`reject invalid primary percentage: ${percentage}`, async (t) => {
    const invalid = { ...prediction, fasttext_confidence_percentage: percentage };
    const { serviceUrl } = await serve(t, (_request, response) => response.end(JSON.stringify(invalid)));
    await assert.rejects(createLanguageClient({ serviceUrl }).predict("ubo"), { code: "INVALID_RESPONSE" });
  });
}

test("reject a batch result missing requested candidates", async (t) => {
  const { serviceUrl } = await serve(t, (_request, response) => response.end(JSON.stringify({
    results: [twoLanguagePrediction, prediction],
  })));
  await assert.rejects(createLanguageClient({ serviceUrl }).predictBatch(["a", "b"], { topK: 2 }), {
    code: "INVALID_RESPONSE",
  });
});

for (const status of [422, 500, 503]) {
  test(`HTTP ${status} is exposed without retries`, async (t) => {
    let calls = 0;
    const { serviceUrl } = await serve(t, (request, response) => {
      calls += 1;
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ detail: "service error" }));
    });
    await assert.rejects(createLanguageClient({ serviceUrl }).predict("ubo"), (error) => {
      assert.ok(error instanceof LanguageClientError);
      assert.equal(error.code, "HTTP_ERROR");
      assert.equal(error.status, status);
      assert.deepEqual(error.details, { detail: "service error" });
      return true;
    });
    assert.equal(calls, 1);
  });
}

for (const body of ["not JSON", "{}", JSON.stringify({ ...prediction, fasttext_confidence: 2 })]) {
  test(`reject malformed success: ${body}`, async (t) => {
    const { serviceUrl } = await serve(t, (request, response) => response.end(body));
    await assert.rejects(createLanguageClient({ serviceUrl }).predict("ubo"), { code: "INVALID_RESPONSE" });
  });
}

test("reject an incomplete batch response", async (t) => {
  const { serviceUrl } = await serve(t, (request, response) => response.end('{"results":[]}'));
  await assert.rejects(createLanguageClient({ serviceUrl }).predictBatch(["ubo"]), { code: "INVALID_RESPONSE" });
});

test("timeout covers a stalled response body", async (t) => {
  const { serviceUrl } = await serve(t, (request, response) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.write('{"language":');
  });
  await assert.rejects(createLanguageClient({ serviceUrl, timeoutMs: 50 }).predict("ubo"), { code: "TIMEOUT" });
});

test("connection failures have a distinct error code", async (t) => {
  const { server, serviceUrl } = await serve(t, () => {});
  await new Promise((resolve) => server.close(resolve));
  await assert.rejects(createLanguageClient({ serviceUrl }).predict("ubo"), { code: "CONNECTION_ERROR" });
});

test("real service single, batch, and validation", {
  skip: process.env.HEALTHPH_TEST_REAL_SERVICE !== "1",
}, async () => {
  const client = createLanguageClient();
  const text = "ubo at lagnat";
  const single = await client.predict(text);
  const { results } = await client.predictBatch([text, "feeling okay", text]);
  assert.equal(single.language, "hil");
  assert.equal(single.prediction_source, "keyword_override");
  assert.equal(results.length, 3);
  for (const index of [0, 2]) assert.deepEqual(results[index], single);
  const mixedText = "Hello po, can I ask kung saan ang clinic?";
  const mixed = await client.predict(mixedText, { topK: 2 });
  assert.deepEqual(mixed.fasttext_predictions.map((candidate) => candidate.language), ["tl", "en"]);
  assert.ok(mixed.fasttext_predictions[1].confidence >= 0.3);
  const mixedBatch = await client.predictBatch([mixedText, text, mixedText], { topK: 2 });
  assert.deepEqual(mixedBatch.results[0], mixed);
  assert.deepEqual(mixedBatch.results[2], mixed);
  const filtered = await client.predict(mixedText, { topK: 2, secondaryMinConfidence: 1 });
  assert.deepEqual(filtered.fasttext_predictions, mixed.fasttext_predictions.slice(0, 1));
  await assert.rejects(client.predict(mixedText, { topK: 3 }), { code: "HTTP_ERROR", status: 422 });
  await assert.rejects(client.predict(" "), { code: "HTTP_ERROR", status: 422 });
  const { stdout } = await promisify(execFile)(process.execPath, [
    fileURLToPath(new URL("../examples/language-client.mjs", import.meta.url)), text,
  ], { env: process.env, timeout: 35_000 });
  const cliResult = JSON.parse(stdout);
  assert.deepEqual(cliResult, single);
});
