import { isDiveyeEnabled, scoreWithDiveye } from "./diveyeClient.js";
import { isNodeFallbackEnabled, scoreWithNodeFallback } from "./aiFallbackDetector.js";
import { isRadarEnabled, scoreWithRadar } from "./radarClient.js";

export function isLocalModelEnabled() {
  // Enable if either DivEye, RADAR, or the local node fallback is active
  if (process.env.AI_LOCAL_MODEL_ENABLED === "false") return false;
  return isDiveyeEnabled() || isRadarEnabled() || isNodeFallbackEnabled();
}

export async function scoreWithLocalModel(text, timeoutMs = 60000) {
  if (!isLocalModelEnabled()) {
    throw new Error("local_model_disabled");
  }

  // Try RADAR first (if enabled)
  if (isRadarEnabled()) {
      try {
          return await scoreWithRadar(text, timeoutMs);
      } catch (error) {
          console.warn("[AI] RADAR unavailable/failed:", error instanceof Error ? error.message : String(error));
      }
  }

  // Try DivEye next (if enabled)
  if (isDiveyeEnabled()) {
    try {
      return await scoreWithDiveye(text, timeoutMs);
    } catch (error) {
      // DivEye failed — fall through to local node fallback below
      console.warn("[AI] DivEye unavailable:", error instanceof Error ? error.message : String(error));
    }
  }

  // Use local node fallback (ZipPy compression model or heuristic)
  if (!isNodeFallbackEnabled()) {
    throw new Error("all_detectors_failed_and_fallback_disabled");
  }

  const fallback = await scoreWithNodeFallback(text);
  return {
    ...fallback,
    source: fallback?.source || "zippy_node_fallback",
  };
}
