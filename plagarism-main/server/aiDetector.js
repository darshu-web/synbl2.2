import { isLocalModelEnabled, scoreWithLocalModel } from "./aiModelClient.js";

const MIN_WORDS_FOR_CONFIDENT_RESULT = 15;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cleanText(text) {
  return (typeof text === "string" ? text : String(text || ""))
    .replace(/\s+/g, " ")
    .trim();
}

function toPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  if (numeric <= 1) return numeric * 100;
  return numeric;
}

function classifyAiScore(aiProbability, votes) {
  const aiVotes = Number(votes?.ai || 0);
  const humanVotes = Number(votes?.human || 0);
  const totalVotes = aiVotes + humanVotes;

  if (totalVotes > 0) {
    if (aiVotes === totalVotes) return "likely_ai";
    if (humanVotes === totalVotes) return "likely_human";

    if (aiVotes - humanVotes >= 2) return "likely_ai";
    if (humanVotes - aiVotes >= 2) return "likely_human";
  }

  if (aiProbability >= 70) return "likely_ai";
  if (aiProbability <= 30) return "likely_human";
  return "mixed_or_uncertain";
}

function createProviderResult({
  id,
  label,
  status,
  score = null,
  confidence = null,
  reason = null,
}) {
  return {
    id,
    label,
    status,
    score: score == null ? null : Math.round(score),
    confidence: confidence == null ? null : Math.round(confidence),
    reason,
  };
}

function summarize(classification, aiProbability, wordCount, detectorName = "AI detector") {
  if (classification === "insufficient_text") {
    return `${detectorName} scored the sample, but ${wordCount} words is too short for a high-confidence result.`;
  }

  if (classification === "likely_ai") {
    return `${detectorName} indicates likely AI-generated content (${aiProbability}% AI probability).`;
  }

  if (classification === "mixed_or_uncertain") {
    return `${detectorName} produced mixed signals (${aiProbability}% AI probability).`;
  }

  return `${detectorName} indicates likely human-written content (${aiProbability}% AI probability).`;
}

function buildEngineProviders(engines, overallConfidence, enginePrefix = "AI") {
  if (!Array.isArray(engines)) return [];

  const safePrefix = String(enginePrefix || "ai")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_");

  return engines.map((engine) => {
    const engineName = String(engine?.engine || "unknown").toLowerCase();
    const score = clamp(toPercent(engine?.ai_probability), 0, 100);

    return createProviderResult({
      id: `${safePrefix}_${engineName}`,
      label: `${enginePrefix} ${engineName.toUpperCase()}`,
      status: "ok",
      score,
      confidence: overallConfidence,
    });
  });
}

function resolveProviderMeta(rawResult) {
  const source = String(rawResult?.source || rawResult?.model || "").toLowerCase();

  if (source.includes("diveye")) {
    return {
      id: "diveye_space",
      label: "DivEye (Hugging Face Space)",
      enginePrefix: "DivEye",
    };
  }

  if (source.includes("zippy")) {
    return {
      id: "zippy_fallback",
      label: "ZipPy Fallback (Local)",
      enginePrefix: "ZipPy",
    };
  }

  if (source.includes("radar")) {
    return {
      id: "radar_local",
      label: "RADAR (Local HuggingFace Model)",
      enginePrefix: "RADAR",
    };
  }

  if (source.includes("heuristic")) {
    return {
      id: "heuristic_fallback",
      label: "Heuristic Fallback (Local)",
      enginePrefix: "Heuristic",
    };
  }

  return {
    id: "ai_detector",
    label: "AI Detector",
    enginePrefix: "AI",
  };
}

function buildIndicators(result) {
  const signedScore = Number(result?.signed_score);
  const aiVotes = Number(result?.votes?.ai || 0);
  const humanVotes = Number(result?.votes?.human || 0);
  const totalVotes = aiVotes + humanVotes;

  return [
    {
      name: "Model Decision Margin",
      value: Number.isFinite(signedScore) ? Number(signedScore.toFixed(3)) : null,
      note:
        Number.isFinite(signedScore) && signedScore < 0
          ? "Model leaned AI"
          : "Model leaned human",
    },
    {
      name: "Engine AI Votes",
      value: aiVotes,
      note: totalVotes > 0 ? `${aiVotes}/${totalVotes} engines voted AI` : "No votes available",
    },
  ];
}

function buildMetrics(text, result) {
  const words = text ? text.split(/\s+/).filter(Boolean).length : 0;

  return {
    tokenCount: words,
    sentenceCount: (text.match(/[^.!?]+[.!?]*/g) || []).filter((s) => s.trim()).length,
    textLength: text.length,
    signedScore: Number.isFinite(Number(result?.signed_score))
      ? Number(result.signed_score)
      : null,
    votes: {
      ai: Number(result?.votes?.ai || 0),
      human: Number(result?.votes?.human || 0),
    },
    engineCount: Array.isArray(result?.engines) ? result.engines.length : 0,
  };
}

export async function detectAIContent(text) {
  const safeText = cleanText(text);

  if (!safeText) {
    return {
      aiProbability: 0,
      classification: "insufficient_text",
      confidence: 0,
      summary:
        "No text was provided for AI detection. Paste text or upload a PDF with extractable text.",
      providers: [
        createProviderResult({
          id: "diveye_space",
          label: "DivEye (Hugging Face Space)",
          status: "skipped",
          reason: "empty_text",
        }),
      ],
      indicators: [],
      metrics: {
        tokenCount: 0,
        sentenceCount: 0,
        textLength: 0,
      },
    };
  }

  if (!isLocalModelEnabled()) {
    return {
      aiProbability: 0,
      classification: "insufficient_text",
      confidence: 0,
      summary: "DivEye detection is disabled by environment configuration.",

      providers: [
        createProviderResult({
          id: "diveye_space",
          label: "DivEye (Hugging Face Space)",
          status: "skipped",
          reason: "local_model_disabled",
        }),
      ],
      indicators: [],
      metrics: buildMetrics(safeText, {}),
    };
  }

  try {
    const rawResult = await scoreWithLocalModel(safeText, 45000);
    const aiProbability = clamp(
      Math.round(toPercent(rawResult?.fake_probability)),
      0,
      100
    );

    const fallbackConfidence = Math.max(35, 100 - Math.abs(aiProbability - 50));
    const confidence = clamp(
      Math.round(toPercent(rawResult?.confidence ?? fallbackConfidence)),
      0,
      100
    );

    const wordCount = safeText.split(/\s+/).filter(Boolean).length;
    const baseClassification = classifyAiScore(aiProbability, rawResult?.votes);
    const classification =
      wordCount < MIN_WORDS_FOR_CONFIDENT_RESULT
        ? "insufficient_text"
        : baseClassification;
    const providerMeta = resolveProviderMeta(rawResult);
    const fallbackSuffix = rawResult?.fallback_reason
      ? ` DivEye unavailable, using ${providerMeta.enginePrefix} fallback.`
      : "";

    const providers = [
      createProviderResult({
        id: providerMeta.id,
        label: providerMeta.label,
        status: "ok",
        score: aiProbability,
        confidence,
      }),
      ...buildEngineProviders(
        rawResult?.engines,
        confidence,
        providerMeta.enginePrefix
      ),
    ];

    return {
      aiProbability,
      classification,
      confidence,
      summary: summarize(
        classification,
        aiProbability,
        wordCount,
        providerMeta.enginePrefix
      ) + fallbackSuffix,
      providers,
      indicators: buildIndicators(rawResult),
      metrics: buildMetrics(safeText, rawResult),
    };
  } catch (error) {
    return {
      aiProbability: 0,
      classification: "insufficient_text",
      confidence: 0,
      summary:
        "DivEye detection failed. Verify internet access and the Diveye Space endpoint configuration.",
      providers: [
        createProviderResult({
          id: "diveye_space",
          label: "DivEye (Hugging Face Space)",
          status: "error",
          reason: error instanceof Error ? error.message : "diveye_detection_failed",
        }),
      ],
      indicators: [],
      metrics: buildMetrics(safeText, {}),
    };
  }
}
