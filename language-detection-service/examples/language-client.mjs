import { pathToFileURL } from "node:url";

export class LanguageClientError extends Error {
  constructor(message, { code, status, details, cause } = {}) {
    super(message, { cause });
    this.name = "LanguageClientError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

function isPrediction(value) {
  return value && typeof value.language === "string" && value.language.length > 0
    && typeof value.cleaned_text === "string" && value.cleaned_text.length > 0
    && typeof value.is_supported === "boolean"
    && ["fasttext", "keyword_override"].includes(value.prediction_source)
    && typeof value.fasttext_language === "string" && value.fasttext_language.length > 0
    && Number.isFinite(value.fasttext_confidence)
    && value.fasttext_confidence >= 0 && value.fasttext_confidence <= 1;
}

export function createLanguageClient({
  serviceUrl = process.env.HEALTHPH_LANGUAGE_URL ?? "http://127.0.0.1:8001",
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
        throw new LanguageClientError(`Language service returned HTTP ${response.status}`, {
          code: "HTTP_ERROR", status: response.status, details: data ?? text,
        });
      }
      if (!validateResponse(data)) {
        throw new LanguageClientError("Language service returned an invalid response", {
          code: "INVALID_RESPONSE", status: response.status,
        });
      }
      return data;
    } catch (error) {
      if (error instanceof LanguageClientError) throw error;
      if (controller.signal.aborted) {
        throw new LanguageClientError(`Language request timed out after ${timeoutMs} ms`, {
          code: "TIMEOUT", cause: error,
        });
      }
      throw new LanguageClientError("Could not connect to or read from the language service", {
        code: "CONNECTION_ERROR", cause: error,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    predict(text) {
      return post("/v1/language/predict", { text }, isPrediction);
    },
    predictBatch(texts) {
      return post("/v1/language/predict-batch", { texts }, (data) =>
        data && Array.isArray(data.results) && data.results.length === texts.length
        && data.results.every(isPrediction));
    },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const text = process.argv.slice(2).join(" ") || "ubo at lagnat";
  try {
    console.log(JSON.stringify(await createLanguageClient().predict(text), null, 2));
  } catch (error) {
    console.error(`${error.code ?? error.name}: ${error.message}`);
    process.exitCode = 1;
  }
}
