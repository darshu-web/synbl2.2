import { spawn } from "child_process";
import { randomUUID } from "crypto";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT_PATH = path.join(__dirname, "radar_service.py");

let childProcess = null;
let isReady = false;
let initPromise = null;
const pendingRequests = new Map();

function startRadarService() {
  if (initPromise) return initPromise;

  initPromise = new Promise((resolve, reject) => {
    console.log("[RADAR] Starting Python service. (Note: May take a moment to download weights on first run)...");
    
    // Determine the python executable. Fallback to just "python" if not specified.
    const pythonExec = process.env.PYTHON_PATH || "python";
    childProcess = spawn(pythonExec, [SCRIPT_PATH], { stdio: ["pipe", "pipe", "inherit"] });

    childProcess.stdout.on("data", (data) => {
      const messages = data.toString().split("\n").filter(Boolean);
      for (const msg of messages) {
        try {
          const parsed = JSON.parse(msg);
          
          if (parsed.type === "ready") {
            console.log(`[RADAR] Model loaded successfully: ${parsed.model} on ${parsed.device}`);
            isReady = true;
            resolve();
          } else if (parsed.type === "fatal") {
            console.error(`[RADAR] Fatal error: ${parsed.error}`);
            reject(new Error(parsed.error));
          } else if (parsed.type === "result") {
            const req = pendingRequests.get(parsed.id);
            if (req) {
              if (parsed.ok) {
                req.resolve(parsed);
              } else {
                req.reject(new Error(parsed.error));
              }
              pendingRequests.delete(parsed.id);
            }
          }
        } catch (e) {
            // Some logs coming from transformers might not be JSON, just pipe them to stdout
            console.log(`[RADAR_LOG]: ${msg}`);
        }
      }
    });

    childProcess.on("close", (code) => {
      console.warn(`[RADAR] Process exited with code ${code}`);
      isReady = false;
      childProcess = null;
      initPromise = null;
      
      // Reject any pending requests
      for (const req of pendingRequests.values()) {
        req.reject(new Error("RADAR service closed unexpectedly"));
      }
      pendingRequests.clear();
      
      if (!isReady) {
          reject(new Error(`RADAR service died before ready with code ${code}`));
      }
    });
    
    childProcess.on("error", (error) => {
        console.error(`[RADAR] Failed to start python process: ${error.message}`);
        isReady = false;
        childProcess = null;
        initPromise = null;
        reject(error);
    });
  });

  return initPromise;
}

export function isRadarEnabled() {
  return process.env.AI_RADAR_ENABLED === "true";
}

export async function scoreWithRadar(text, timeoutMs = 120000) {
  if (!isRadarEnabled()) {
      throw new Error("radar_model_disabled");
  }

  // Ensure service is running and model is loaded
  await startRadarService();

  return new Promise((resolve, reject) => {
    const id = randomUUID();
    
    const timeout = setTimeout(() => {
      pendingRequests.delete(id);
      reject(new Error("RADAR request timed out"));
    }, timeoutMs);

    pendingRequests.set(id, {
      resolve: (val) => {
        clearTimeout(timeout);
        // Map the result format expected by aiDetector.js
        resolve({
            fake_probability: val.fake_probability,
            real_probability: val.real_probability,
            confidence: val.confidence,
            signed_score: val.signed_score,
            votes: val.votes,
            engines: val.engines,
            source: "radar_local",
        });
      },
      reject: (err) => {
        clearTimeout(timeout);
        reject(err);
      }
    });

    try {
      childProcess.stdin.write(JSON.stringify({ id, text }) + "\n");
    } catch (err) {
      clearTimeout(timeout);
      pendingRequests.delete(id);
      reject(err);
    }
  });
}
