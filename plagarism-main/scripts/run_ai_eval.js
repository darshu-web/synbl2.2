import { config } from "dotenv";
config();

import { detectAIContent } from "../server/aiDetector.js";
const sampleSpecs = [
  {
    id: "human_1",
    expected: "human",
    text:
      "I reached the office late because of heavy traffic, then spent the morning fixing a spreadsheet formula and replying to customer emails. After lunch I joined a planning call and wrote down the action items for next week.",
  },
  {
    id: "human_2",
    expected: "human",
    text:
      "Last weekend we cleaned the balcony, repotted two plants, and cooked dinner together at home. It rained in the evening, so we stayed in, watched a movie, and finished the leftover dessert.",
  },
  {
    id: "human_3",
    expected: "human",
    text:
      "My cousin borrowed my bike this morning, so I walked to the store instead. On the way back I met a neighbor, and we spoke for a few minutes about the road repairs happening near the market.",
  },
  {
    id: "human_4",
    expected: "human",
    text:
      "I reviewed my notes from yesterday's class, highlighted the sections I did not understand, and sent questions to the instructor. In the evening I practiced those concepts again with a friend over a video call.",
  },
  {
    id: "human_5",
    expected: "human",
    text:
      "We had a family lunch today with simple food and a lot of conversation about travel plans. Later I sorted old photos on my laptop and backed up important files to avoid losing them.",
  },
  {
    id: "gpt_1",
    expected: "ai",
    text:
      "Artificial intelligence systems generate fluent text by estimating token probabilities across large corpora and decoding outputs through controlled sampling strategies. Their utility spans summarization, drafting, and conversational assistance in enterprise environments.",
  },
  {
    id: "gpt_2",
    expected: "ai",
    text:
      "In modern software engineering, scalability emerges from architectural modularity, observability-first design, and resilient failure handling. Teams that align delivery pipelines with measurable reliability objectives consistently reduce operational risk while accelerating iteration speed.",
  },
  {
    id: "gpt_3",
    expected: "ai",
    text:
      "Climate adaptation frameworks must integrate infrastructure resilience, early-warning systems, and inclusive governance to mitigate long-term socioeconomic disruption. Data-driven planning enables policymakers to allocate resources efficiently under uncertain environmental conditions.",
  },
  {
    id: "gpt_4",
    expected: "ai",
    text:
      "The transformer architecture improves sequence modeling by combining self-attention with positional representations, enabling richer contextual reasoning than recurrent approaches. As model scale increases, emergent capabilities can appear across diverse linguistic and analytical tasks.",
  },
  {
    id: "gpt_5",
    expected: "ai",
    text:
      "Effective cybersecurity programs layer identity controls, continuous monitoring, and incident response automation to reduce mean time to detection and recovery. Mature organizations pair technical safeguards with regular simulation exercises and governance oversight.",
  },
];

function toPrediction(classification) {
  if (classification === "likely_ai") return "ai";
  if (classification === "likely_human") return "human";
  return "uncertain";
}

async function main() {
  const rows = [];

  for (const sample of sampleSpecs) {
    const result = await detectAIContent(sample.text);
    const prediction = toPrediction(result.classification);
    rows.push({
      id: sample.id,
      expected: sample.expected,
      prediction,
      aiProbability: Math.round(Number(result.aiProbability || 0)),
      confidence: Math.round(Number(result.confidence || 0)),
      classification: result.classification,
      provider: result?.providers?.[0]?.id || "unknown",
      correct: prediction === sample.expected,
    });
  }

  const decided = rows.filter((row) => row.prediction !== "uncertain");
  const correct = rows.filter((row) => row.correct);
  const decidedCorrect = decided.filter((row) => row.correct);

  console.log("DivEye 10-Sample Evaluation");
  console.log("---------------------------");
  for (const row of rows) {
    console.log(
      `${row.id.padEnd(8)} expected=${row.expected.padEnd(5)} predicted=${row.prediction.padEnd(9)} ai=${String(row.aiProbability).padStart(3)}% confidence=${String(row.confidence).padStart(3)} class=${row.classification}`
        + ` provider=${row.provider}`
    );
  }
  console.log("---------------------------");
  console.log(`Overall accuracy: ${((correct.length / rows.length) * 100).toFixed(1)}% (${correct.length}/${rows.length})`);
  console.log(
    `Decided-only accuracy: ${(
      decided.length ? (decidedCorrect.length / decided.length) * 100 : 0
    ).toFixed(1)}% (${decidedCorrect.length}/${decided.length})`
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
