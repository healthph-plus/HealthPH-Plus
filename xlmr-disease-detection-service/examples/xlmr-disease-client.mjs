import { pathToFileURL } from "node:url";

const LABELS = ["AURI", "PN", "TB", "COVID"];

export class XlmrDiseaseClientError extends Error {
  constructor(message, { code, status, details, cause } = {}) {
    super(message, { cause });
    this.name = "XlmrDiseaseClientError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

function isScoreMap(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).length === LABELS.length && LABELS.every(
    (label) => Number.isFinite(value[label]) && value[label] >= 0 && value[label] <= 1,
  );
}

function validSymptoms(symptoms, text) {
  if (!Array.isArray(symptoms) || typeof text !== "string") return false;
  const characters = Array.from(text); // Python offsets count Unicode code points.
  const keys = ["text", "normalized_symptom", "start_index", "end_index", "confidence", "negated"];
  let previousEnd = 0;
  return symptoms.every((symptom) => {
    if (!symptom || typeof symptom !== "object" || Array.isArray(symptom)
      || Object.keys(symptom).length !== keys.length
      || !keys.every((key) => Object.hasOwn(symptom, key))
      || typeof symptom.text !== "string" || symptom.text.length === 0
      || !(symptom.normalized_symptom === null
        || (typeof symptom.normalized_symptom === "string" && symptom.normalized_symptom.length > 0))
      || !Number.isSafeInteger(symptom.start_index) || !Number.isSafeInteger(symptom.end_index)
      || symptom.start_index < previousEnd || symptom.end_index <= symptom.start_index
      || symptom.end_index > characters.length
      || !Number.isFinite(symptom.confidence) || symptom.confidence < 0 || symptom.confidence > 1
      || typeof symptom.negated !== "boolean"
      || characters.slice(symptom.start_index, symptom.end_index).join("") !== symptom.text) return false;
    previousEnd = symptom.end_index;
    return true;
  });
}

function isPrediction(value, text) {
  return value && Array.isArray(value.labels)
    && value.labels.every((label) => LABELS.includes(label))
    && new Set(value.labels).size === value.labels.length
    && isScoreMap(value.probabilities) && isScoreMap(value.thresholds_used)
    && JSON.stringify(value.labels) === JSON.stringify(LABELS.filter(
      (label) => value.probabilities[label] >= value.thresholds_used[label],
    ))
    && typeof value.model_version === "string" && value.model_version.length > 0
    && typeof value.symptom_model_version === "string" && value.symptom_model_version.length > 0
    && validSymptoms(value.symptoms, text);
}

export function createXlmrDiseaseClient({
  serviceUrl = process.env.HEALTHPH_XLMR_DISEASE_URL ?? "http://127.0.0.1:8003",
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
        throw new XlmrDiseaseClientError(`Disease service returned HTTP ${response.status}`, {
          code: "HTTP_ERROR", status: response.status, details: data ?? text,
        });
      }
      if (!validateResponse(data)) {
        throw new XlmrDiseaseClientError("Disease service returned an invalid response", {
          code: "INVALID_RESPONSE", status: response.status,
        });
      }
      return data;
    } catch (error) {
      if (error instanceof XlmrDiseaseClientError) throw error;
      if (controller.signal.aborted) {
        throw new XlmrDiseaseClientError(`Disease request timed out after ${timeoutMs} ms`, {
          code: "TIMEOUT", cause: error,
        });
      }
      throw new XlmrDiseaseClientError("Could not connect to or read from the disease service", {
        code: "CONNECTION_ERROR", cause: error,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    predict(text) {
      return post("/v1/disease/predict", { text }, (data) => isPrediction(data, text));
    },
    predictBatch(texts) {
      return post("/v1/disease/predict-batch", { texts }, (data) =>
        data && Array.isArray(data.results) && data.results.length === texts.length
        && data.results.every((result, index) => isPrediction(result, texts[index])));
    },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const text = process.argv.slice(2).join(" ") || "ubo at lagnat";
  try {
    console.log(JSON.stringify(await createXlmrDiseaseClient().predict(text), null, 2));
  } catch (error) {
    console.error(`${error.code ?? error.name}: ${error.message}`);
    process.exitCode = 1;
  }
}
