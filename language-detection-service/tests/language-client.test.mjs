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
  await assert.rejects(client.predict(" "), { code: "HTTP_ERROR", status: 422 });
  const { stdout } = await promisify(execFile)(process.execPath, [
    fileURLToPath(new URL("../examples/language-client.mjs", import.meta.url)), text,
  ], { env: process.env, timeout: 35_000 });
  const cliResult = JSON.parse(stdout);
  assert.deepEqual(cliResult, single);
});
