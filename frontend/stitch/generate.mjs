/**
 * One-off: generate Sentinel Guard dashboard via @google/stitch-sdk
 * Reads STITCH_API_KEY from env or .cursor/mcp.json
 */
import { stitch } from "@google/stitch-sdk";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "../..");
const outDir = path.join(root, "frontend", "stitch");

function loadApiKey() {
  if (process.env.STITCH_API_KEY) return process.env.STITCH_API_KEY;
  const mcpPath = path.join(root, ".cursor", "mcp.json");
  const mcp = JSON.parse(fs.readFileSync(mcpPath, "utf8"));
  return mcp?.mcpServers?.stitch?.env?.STITCH_API_KEY;
}

const apiKey = loadApiKey();
if (!apiKey) {
  console.error("Missing STITCH_API_KEY");
  process.exit(1);
}
process.env.STITCH_API_KEY = apiKey;

fs.mkdirSync(outDir, { recursive: true });

const prompt = `Desktop security operations dashboard. Brand wordmark SENTINEL GUARD. Dark charcoal graphite background with muted teal, amber, and crimson accents. No purple. Top bar with brand and LIVE indicator. Large verdict status strip for NORMAL / STRESS / HARASSMENT. Four metric tiles: Face Emotion, Voice Tone, Text Class, Final Result. Left prediction log table. Right donut chart and proof evidence gallery with face thumbnail and audio player. Distinctive geometric sans typography, monospace for data. Operational refined UI for multimodal harassment detection. Sample data: angry / angry tone / harassment.`;

console.log("Generating Stitch screen…");
const project = stitch.project("9583305098157881670");
const screen = await project.generate(prompt, "DESKTOP", "GEMINI_3_FLASH");
console.log("screen:", screen.id);

const htmlUrl = await screen.getHtml();
const imageUrl = await screen.getImage();
fs.writeFileSync(
  path.join(outDir, "meta.json"),
  JSON.stringify({ screenId: screen.id, htmlUrl, imageUrl }, null, 2)
);

if (htmlUrl) {
  const html = await (await fetch(htmlUrl)).text();
  fs.writeFileSync(path.join(outDir, "dashboard.html"), html);
  console.log("HTML saved:", html.length, "chars");
}
if (imageUrl) {
  const buf = Buffer.from(await (await fetch(imageUrl)).arrayBuffer());
  fs.writeFileSync(path.join(outDir, "dashboard.png"), buf);
  console.log("PNG saved:", buf.length, "bytes");
}
console.log("DONE →", outDir);
