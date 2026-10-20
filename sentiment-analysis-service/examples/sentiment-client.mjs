import { pathToFileURL } from "node:url";

export class SentimentClientError extends Error {
  constructor(message, { code, status, details, cause } = {}) {
    super(message, { cause });
    this.name = "SentimentClientError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

function isProbability(value) {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

function isPrediction(value) {
  if (!value || ![0, 1].includes(value.predicted_label)
      || !isProbability(value.worried_probability) || !isProbability(value.threshold_used)
      || typeof value.model_version !== "string" || value.model_version.length === 0) return false;
  const label = Number(value.worried_probability >= value.threshold_used);
  return value.predicted_label === label && value.predicted_label_name === ["calm", "worried"][label];
}

export function createSentimentClient({
  serviceUrl = process.env.HEALTHPH_SENTIMENT_URL ?? "http://127.0.0.1:8002",
  timeoutMs = 30_000,
} = {}) {
  const base = new URL(serviceUrl);
  if (!["http:", "https:"].includes(base.protocol)) {
    throw new TypeError("serviceUrl must use HTTP or HTTPS");
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2_147_483_647) {
    throw new TypeError("timeoutMs must be a positive 32-bit integer");
  }

  async function post(path, body, validateResponse) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(new URL(path, base), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await response.text();
      let data;
      try { data = JSON.parse(text); } catch { /* Report HTTP or protocol error below. */ }
      if (!response.ok) {
        throw new SentimentClientError(`Sentiment service returned HTTP ${response.status}`, {
          code: "HTTP_ERROR", status: response.status, details: data ?? text,
        });
      }
      if (!validateResponse(data)) {
        throw new SentimentClientError("Sentiment service returned an invalid response", {
          code: "INVALID_RESPONSE", status: response.status,
        });
      }
      return data;
    } catch (error) {
      if (error instanceof SentimentClientError) throw error;
      if (controller.signal.aborted) {
        throw new SentimentClientError(`Sentiment request timed out after ${timeoutMs} ms`, {
          code: "TIMEOUT", cause: error,
        });
      }
      throw new SentimentClientError("Could not connect to or read from the sentiment service", {
        code: "CONNECTION_ERROR", cause: error,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    predict(text) {
      return post("/v1/sentiment/predict", { text }, isPrediction);
    },
    predictBatch(texts) {
      return post("/v1/sentiment/predict-batch", { texts }, (data) =>
        data && Array.isArray(data.results) && data.results.length === texts.length
        && data.results.every(isPrediction));
    },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const text = process.argv.slice(2).join(" ") || "ubo at lagnat";
  try {
    console.log(JSON.stringify(await createSentimentClient().predict(text), null, 2));
  } catch (error) {
    console.error(`${error.code ?? error.name}: ${error.message}`);
    process.exitCode = 1;
  }
}
