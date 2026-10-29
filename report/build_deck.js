// Builds report/ML_Markowitz_presentation.pptx (7 slides). Run from report/: npm install pptxgenjs; node build_deck.js
// Charts are native PowerPoint charts filled from results_summary.json; equations are LaTeX-rendered PNGs (eq/).
// Optional: APPLY_THEME=<path to an apply_theme.js> also writes the colour theme into the file.
const fs = require("fs");
const path = require("path");
const pptxgen = require(process.env.PPTXGEN || "pptxgenjs");
const applyTheme = process.env.APPLY_THEME ? require(process.env.APPLY_THEME).applyTheme : null;

const OUT = path.join(__dirname, "ML_Markowitz_presentation.pptx");
const THEME = {
  name: "Decision Learning",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2A44", lt1: "FFFFFF", dk2: "0E5E6F", lt2: "EEF2F7",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "0E5E6F", accent6: "6250D6", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { navy: "1F2A44", teal: "0E5E6F", orange: "EB6834", blue: "2A78D6", muted: "898781",
  ink2: "52514E", grid: "E1E0D9", tintBlue: "EEF2F7", tintTeal: "E2EFF0", tintOrange: "FCEBE3", white: "FFFFFF" };

const S = JSON.parse(fs.readFileSync(path.join(__dirname, "results_summary.json"), "utf8"));
const eqSize = Object.fromEntries(fs.readFileSync(path.join(__dirname, "eq", "sizes.txt"), "utf8").trim().split("\n")
  .map((l) => { const [k, w, h] = l.split(" "); return [k, [parseFloat(w), parseFloat(h)]]; }));

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Learning for the Decision";
pres.author = "Kumar Utkarsh";
const C = pres.SchemeColor;
const W = 13.333;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 7.4, h: 1.5, fontSize: 44, bold: true,
      color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.7, w: 7.0, h: 1.6, fontSize: 18,
      color: "CADCFC", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.white },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.85, fontSize: 32, bold: true,
      color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Machine Learning Meets Markowitz: replication and extension", options: { x: 0.6, y: 7.02, w: 8, h: 0.3,
      fontSize: 10, color: HEX.muted, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
});

// ---------- helpers ----------
const box = (slide, x, y, w, h, fill, name, radius = 0.08) =>
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: radius, objectName: name });
const text = (slide, t, o) => slide.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });
const circle = (slide, x, y, d, fill, label, name) => {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });
  text(slide, label, { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, fontSize: d > 0.6 ? 18 : 13, color: C.background1 });
};
const arrow = (slide, x1, y1, x2, y2, color, dash, name) =>
  slide.addShape(pres.ShapeType.line, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 1.75, dashType: dash || "solid", endArrowType: "triangle" }, objectName: name });
const eq = (slide, name, x, y, maxW, maxH, scale = 1.3) => {
  const [w0, h0] = eqSize[name];
  const s = Math.min(maxW / w0, maxH / h0, scale);
  slide.addImage({ path: path.join(__dirname, "eq", `${name}.png`), x, y, w: w0 * s, h: h0 * s, altText: `Equation: ${name}` });
  return h0 * s;
};
const card = (slide, x, y, w, h, fill, heading, name) => {
  box(slide, x, y, w, h, fill, name);
  text(slide, heading, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text1 });
};

// ---------- 1. Title ----------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Learning for the Decision", { placeholder: "title" });
s.addText([
  { text: "Replicating and extending end-to-end Markowitz learning on China A-shares", options: { breakLine: true } },
  { text: "Based on Wang, Gao, Harvey, Liu & Tao (2026), Machine Learning Meets Markowitz", options: { breakLine: true, fontSize: 14, color: "9FB3D9" } },
  { text: "Kumar Utkarsh  ·  IIT Guwahati  ·  October 2026", options: { fontSize: 14, color: "9FB3D9" } },
], { placeholder: "body" });
// motif: predict -> optimize -> decide
const steps = [["1", "Predict", HEX.teal], ["2", "Optimize", HEX.blue], ["3", "Decide", HEX.orange]];
steps.forEach(([n, lab, col], i) => {
  const cx = 8.95 + i * 1.35, cy = 3.05;
  circle(s, cx, cy, 0.95, col, n, `motif-${lab}`);
  text(s, lab, { x: cx - 0.3, y: cy + 1.1, w: 1.55, h: 0.4, align: "center", fontSize: 14, color: C.background1, bold: true });
  if (i < 2) arrow(s, cx + 1.0, cy + 0.47, cx + 1.3, cy + 0.47, "9FB3D9", "solid", `motif-arrow-${i}`);
});
text(s, "Train the forecaster on the portfolio it produces", { x: 8.55, y: 4.75, w: 4.2, h: 0.5, fontSize: 14, italic: true, color: "CADCFC", align: "center" });
s.addNotes("The paper asks whether a return forecaster should be trained for statistical accuracy or for the portfolio it feeds. We replicate it on free China A-share data, audit it, and extend it.");

// ---------- 2. Problem ----------
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("Why predict-then-optimize falls short", { placeholder: "title" });
const flow = (y, label, lossText, lossX, color, e2e) => {
  text(s, label, { x: 0.6, y, w: 6.6, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
  const names = ["Characteristics", "Model g(x; θ)", "Forecast r̂", "Optimizer", "Portfolio w"];
  names.forEach((n, i) => {
    const x = 0.6 + i * 1.34;
    box(s, x, y + 0.45, 1.16, 0.62, i === 3 && e2e ? HEX.tintOrange : HEX.tintBlue, `${label}-${n}`, 0.06);
    text(s, n, { x, y: y + 0.45, w: 1.16, h: 0.62, align: "center", valign: "middle", fontSize: 12 });
    if (i < 4) arrow(s, x + 1.18, y + 0.76, x + 1.32, y + 0.76, HEX.navy, "solid", `${label}-a${i}`);
  });
  box(s, lossX, y + 1.35, 2.6, 0.55, color, `${label}-loss`, 0.06);
  text(s, lossText, { x: lossX, y: y + 1.35, w: 2.6, h: 0.55, align: "center", valign: "middle", fontSize: 12, bold: true, color: C.background1 });
};
flow(1.5, "Two-stage (the benchmark)", "Loss: MSE, every stock equal", 0.6 + 1.34 - 0.72, HEX.teal, false);
arrow(s, 0.6 + 1.34 + 0.58, 1.5 + 1.35, 0.6 + 1.34 + 0.58, 1.5 + 1.09, HEX.teal, "dash", "ts-feedback");
flow(3.75, "End-to-end (the paper)", "Loss: realized portfolio return", 0.6 + 4 * 1.34 - 1.44, HEX.orange, true);
arrow(s, 0.6 + 4 * 1.34 + 0.58, 3.75 + 1.35, 0.6 + 4 * 1.34 + 0.58, 3.75 + 1.09, HEX.orange, "dash", "e2e-up");
arrow(s, 0.6 + 4 * 1.34 - 1.44, 3.75 + 1.62, 0.6 + 1.34 + 0.58, 3.75 + 1.62, HEX.orange, "dash", "e2e-back");
text(s, "gradients flow back through the optimizer", { x: 1.9, y: 5.45, w: 3.4, h: 0.3, fontSize: 11, italic: true, color: HEX.ink2 });
// right: the paper's four-asset example, reproduced
card(s, 7.55, 1.5, 5.2, 4.55, HEX.tintTeal, "The paper's example, reproduced exactly", "example-card");
text(s, "Realized utility (%) of mean-variance portfolios built from three pricing models", { x: 7.8, y: 2.05, w: 4.7, h: 0.55, fontSize: 12, color: HEX.ink2 });
const hdr = { bold: true, color: C.text1, fill: { color: "CFE3E5" }, fontSize: 12, align: "center" };
const cell = (t, best) => ({ text: t, options: { align: "center", fontSize: 13, bold: !!best, color: best ? HEX.orange : C.text1 } });
s.addTable([
  [{ text: "Investor", options: { ...hdr, align: "left" } }, { text: "OLS (best fit)", options: hdr }, { text: "pro-large-cap", options: hdr }, { text: "pro-small-cap", options: hdr }],
  [{ text: "risk-averse, η = 10", options: { fontSize: 12, color: C.text1 } }, cell("31.97"), cell("32.01", true), cell("31.90")],
  [{ text: "risk-tolerant, η = 1", options: { fontSize: 12, color: C.text1 } }, cell("62.84"), cell("62.53"), cell("62.90", true)],
], { x: 7.8, y: 2.75, w: 4.7, colW: [1.55, 1.05, 1.05, 1.05], rowH: 0.45, border: { type: "solid", pt: 0.75, color: "B9D3D6" }, fill: { color: HEX.white } });
text(s, [
  { text: "The model with the lowest MSE is best for neither investor.", options: { bold: true, breakLine: true } },
  { text: "Accuracy matters most where the portfolio puts its weight, and that depends on the investor.", options: {} },
], { x: 7.8, y: 4.35, w: 4.7, h: 1.4, fontSize: 13 });
text(s, "Our check: OLS fit [0.652, 0.927] and all six utilities match the paper (OLS without intercept).", { x: 0.6, y: 6.3, w: 12.1, h: 0.4, fontSize: 12, color: HEX.ink2, italic: true });
s.addNotes("Two-stage learning fits returns for all stocks equally. E2E puts the optimizer in the loop. The paper's four-asset example, which we reproduced exactly, shows the best statistical fit is best for neither investor.");

// ---------- 3. Methodology ----------
pres.addSection({ title: "Methodology" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methodology" });
s.addText("Methodology: learning through the optimizer", { placeholder: "title" });
const left = [
  ["Benchmark: fit, then optimize", "mse", "Linear or NN1–NN5, Adam, early stopping; deciles of r̂ each day"],
  ["Lower level: the portfolio layer", "lower", "Cap u = 1/(10% of N): with ζ and γ at zero this is the equal-weight top decile"],
  ["Upper level: learn θ for the decision", "upper", "ζ smooths the linear program; γ is the transaction cost rate"],
];
left.forEach(([h, e, note], i) => {
  const y = 1.45 + i * 1.78;
  card(s, 0.6, y, 7.2, 1.6, HEX.tintBlue, h, `method-${e}`);
  eq(s, e, 0.85, y + 0.62, 6.7, 0.55);
  text(s, note, { x: 0.85, y: y + 1.2, w: 6.7, h: 0.32, fontSize: 12, color: HEX.ink2 });
});
card(s, 8.1, 1.45, 4.65, 5.34, HEX.tintTeal, "Our exact layer (replaces KKT solvers)", "layer-card");
text(s, "The problem is separable up to the budget, so one threshold τ solves it:", { x: 8.35, y: 2.0, w: 4.15, h: 0.55, fontSize: 12, color: HEX.ink2 });
eq(s, "layer", 8.35, 2.6, 4.15, 0.45);
text(s, "S is a soft threshold around yesterday's weight p. On the free stocks F:", { x: 8.35, y: 3.2, w: 4.15, h: 0.55, fontSize: 12, color: HEX.ink2 });
eq(s, "jacobian", 8.35, 3.8, 4.15, 0.62);
text(s, [
  { text: "O(N) backward pass on the GPU", options: { bullet: true, breakLine: true } },
  { text: "Solution never worse than a generic solver (gap < 1e-17)", options: { bullet: true, breakLine: true } },
  { text: "Gradient matches finite differences (error ~1e-11)", options: { bullet: true } },
], { x: 8.35, y: 4.65, w: 4.15, h: 1.9, fontSize: 13, paraSpaceAfter: 6 });
s.addNotes("The bilevel problem: the inner problem chooses weights from forecasts, the outer problem chooses network parameters for realized performance. Our layer solves the inner problem exactly and gives its Jacobian in closed form.");

// ---------- 4. Machine learning at each stage ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methodology" });
s.addText("Machine learning at each stage, and why", { placeholder: "title" });
const ml = [
  ["Prediction layer", "Linear and NN1–NN5 (32-16-8-4-2); LayerNorm, ReLU, BatchNorm, 5% dropout", "Benchmark architectures of Gu, Kelly & Xiu; small nets resist daily noise"],
  ["Training objective", "MSE for the two-stage model; realized portfolio return for E2E", "Puts accuracy where the long-only portfolio puts its weight"],
  ["Optimization layer", "Smoothed top-decile program with a transaction-cost penalty", "Gradients through argmin; ζ makes the solution Lipschitz"],
  ["Training protocol", "Adam, early stopping (patience 10); E2E warm-started from MSE; 8-day batches", "Stable and fast; days kept in order when costs link them"],
  ["Validation design", "39m / 9m / 3m rolling windows, 2-day embargo, all choices on validation", "No look-ahead: the test quarter is never used to choose anything"],
  ["Our models", "Ridge, PCR, PLS, LDA, GMM experts, SVR, decision-aware PCA", "Regularize 251 collinear features; test reduction chosen by the decision"],
];
ml.forEach(([h, what, why], i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = 0.6 + col * 4.1, y = 1.45 + row * 2.75;
  const fill = i === 5 ? HEX.tintOrange : HEX.tintBlue;
  box(s, x, y, 3.85, 2.5, fill, `ml-${i}`);
  circle(s, x + 0.22, y + 0.2, 0.5, i === 5 ? HEX.orange : HEX.teal, String(i + 1), `ml-num-${i}`);
  text(s, h, { x: x + 0.85, y: y + 0.24, w: 2.85, h: 0.42, fontSize: 16, bold: true });
  text(s, what, { x: x + 0.25, y: y + 0.85, w: 3.4, h: 0.9, fontSize: 13 });
  text(s, [{ text: "Why: ", options: { bold: true, color: C.text2 } }, { text: why, options: { color: HEX.ink2 } }],
    { x: x + 0.25, y: y + 1.75, w: 3.4, h: 0.65, fontSize: 12 });
});
s.addNotes("Each stage uses a specific learning choice. Cards 1 to 5 follow the paper; card 6 lists our additions.");

// ---------- 5. Audit ----------
pres.addSection({ title: "Extensions" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Extensions" });
s.addText("Auditing the paper: what doesn't add up", { placeholder: "title" });
const stats = [
  ["11.2M", "stock-days implied by Table A.1; the text reports 5.96M"],
  ["53.35", "NN3-E2E long-short return implied by its own H and L; 54.9 is reported"],
  ["p = 0.08", "two-sided, for t = 1.74, which the paper marks significant at 5%"],
  ["−0.04%", "2017Q1 top-decile daily return once untradeable stocks are excluded (+0.76% with them)"],
];
stats.forEach(([big, lab], i) => {
  const x = 0.6 + i * 3.1;
  box(s, x, 1.45, 2.85, 1.95, HEX.tintOrange, `stat-${i}`);
  text(s, big, { x: x + 0.22, y: 1.6, w: 2.45, h: 0.8, fontSize: 34, bold: true, color: HEX.orange, fontFace: THEME.headFontFace });
  text(s, lab, { x: x + 0.22, y: 2.45, w: 2.45, h: 0.9, fontSize: 12, color: HEX.ink2 });
});
card(s, 0.6, 3.7, 5.95, 2.6, HEX.tintBlue, "Conventions recovered from the tables", "conv-card");
text(s, [
  { text: "H–L is (H − L)/2 in every cell but one", options: { bullet: true, breakLine: true } },
  { text: "Cost = γ × one-sided turnover ½‖Δw‖₁ (matches Table 3 to 0.2 pp)", options: { bullet: true, breakLine: true } },
  { text: "Annualized with about 243 trading days", options: { bullet: true, breakLine: true } },
  { text: "Drawdowns on cumulative sums; length in calendar days", options: { bullet: true } },
], { x: 0.85, y: 4.25, w: 5.5, h: 2.4, fontSize: 14, paraSpaceAfter: 6 });
card(s, 6.8, 3.7, 5.95, 2.6, HEX.tintTeal, "Our fixes", "fix-card");
text(s, [
  { text: "Smoothing ζ‖w‖² squared and tuned (paper: unsquared, ζ unreported)", options: { bullet: true, breakLine: true } },
  { text: "Daily rebalancing, despite \"quarterly\" captions", options: { bullet: true, breakLine: true } },
  { text: "Two-day embargo at every train / validation / test boundary", options: { bullet: true, breakLine: true } },
  { text: "Tradability: drop new listings and stocks locked at the upper limit", options: { bullet: true, breakLine: true } },
  { text: "Tuned MSE-Opt benchmark; one- and two-sided tests", options: { bullet: true } },
], { x: 7.05, y: 4.25, w: 5.5, h: 2.4, fontSize: 14, paraSpaceAfter: 6 });
s.addNotes("We recomputed the paper's tables. Three numbers do not reconcile, several conventions are implied but unstated, and we adopt explicit fixes for each.");

// ---------- 6. Extensions ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Extensions" });
s.addText("Our extensions: dimension reduction for the decision", { placeholder: "title" });
const pipe = [["251 features", HEX.tintBlue], ["Reduction", HEX.tintOrange], ["Model", HEX.tintBlue], ["Portfolio layer", HEX.tintBlue]];
pipe.forEach(([n, fill], i) => {
  const x = 0.6 + i * 3.1;
  box(s, x, 1.5, 2.7, 0.75, fill, `pipe-${i}`, 0.06);
  text(s, n, { x, y: 1.5, w: 2.7, h: 0.75, align: "center", valign: "middle", fontSize: 15, bold: true });
  if (i < 3) arrow(s, x + 2.75, 1.875, x + 3.05, 1.875, HEX.navy, "solid", `pipe-a${i}`);
});
const ext = [
  ["PCA (unsupervised)", "pcr", "Top-K principal components of the numeric features"],
  ["LDA (supervised by deciles)", "lda", "Directions that separate each day's return deciles"],
  ["Decision-aware PCA (new)", "dapca", "Start at PCA loadings; let the portfolio loss move them"],
  ["GMM mixture of experts", "gmm", "Separate linear rules for large and small firms"],
];
ext.forEach(([h, e, d], i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = 0.6 + col * 4.05, y = 2.6 + row * 2.1;
  card(s, x, y, 3.85, 1.9, i === 2 ? HEX.tintOrange : HEX.tintBlue, h, `ext-${e}`);
  eq(s, e, x + 0.25, y + 0.68, 3.35, 0.5);
  text(s, d, { x: x + 0.25, y: y + 1.32, w: 3.35, h: 0.5, fontSize: 12, color: HEX.ink2 });
});
card(s, 8.85, 2.6, 3.9, 4.0, HEX.tintTeal, "Key point", "insight-card");
text(s, [
  { text: "For a linear model, a learned linear bottleneck adds nothing: (Wγ)ᵀx is still one linear score.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "So reduction must act as a constraint. Decision-aware PCA keeps W near the PCA loadings and lets the portfolio objective move them.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Also: PLS, persistence-aware (low-turnover) components, linear and kernel SVR.", options: { color: HEX.ink2 } },
], { x: 9.1, y: 3.15, w: 3.4, h: 3.3, fontSize: 13 });
s.addNotes("Dimension reduction sits between the features and the model. We compare reduction chosen by variance (PCA), by class separation (LDA) and by the portfolio objective (decision-aware PCA).");

// ---------- 7. Results ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Results: training for the decision (2014–2023)", { placeholder: "title" });
const DF = path.join(__dirname, "results_decision.json");
const D = fs.existsSync(DF) ? JSON.parse(fs.readFileSync(DF, "utf8")) : { pairs: {}, costs: {}, meansd: {} };
const PAIR_LABEL = { "E2E-Linear": "Linear", "E2E-NN3": "NN3", "E2E-Ridge": "Ridge", "E2E-GMM-Experts": "GMM experts",
  "E2E-PCA20-NN3": "PCA20-NN3, PCA fixed", "DA0.1-PCA20-NN3": "PCA20-NN3, DA μ=0.1", "DA1-PCA20-NN3": "PCA20-NN3, DA μ=1",
  "DA0.1-PCA20-Linear": "PCA20-Linear, DA μ=0.1",
  "DA1-PCA20-Linear": "PCA20-Linear, DA μ=1" };
const pairs = Object.keys(PAIR_LABEL).filter((k) => D.pairs[k]).reverse(); // bar charts list bottom-up
const r2 = (v) => Number(v.toFixed(2));
s.addChart(pres.ChartType.bar, [
  { name: "Two-stage (MSE)", labels: pairs.map((k) => PAIR_LABEL[k]), values: pairs.map((k) => r2(D.pairs[k].two_stage.sr)) },
  { name: "End-to-end", labels: pairs.map((k) => PAIR_LABEL[k]), values: pairs.map((k) => r2(D.pairs[k].e2e.sr)) },
], {
  x: 0.6, y: 1.4, w: 6.3, h: 4.45, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.blue, HEX.orange],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, dataLabelColor: HEX.ink2,
  dataLabelFontFace: "+mn-lt", showTitle: true, title: "Long-only top decile: Sharpe ratio, before costs", titleFontSize: 14,
  titleColor: HEX.navy, titleFontFace: "+mn-lt", catAxisLabelColor: HEX.ink2, catAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", catAxisLineShow: false, valAxisHidden: true, valGridLine: { style: "none" },
  catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 11, legendFontFace: "+mn-lt",
  legendColor: HEX.ink2, barGapWidthPct: 40, objectName: "chart-e2e-vs-two-stage",
});
// three findings, each a big number with its source
const stat = (y, fill, big, head, body, name) => {
  box(s, 7.15, y, 5.6, 1.42, fill, name);
  text(s, big, { x: 7.35, y: y + 0.14, w: 1.85, h: 1.15, fontSize: 28, bold: true, color: HEX.orange, valign: "middle", fontFace: THEME.headFontFace });
  text(s, [{ text: head, options: { bold: true, breakLine: true } }, { text: body, options: { color: HEX.ink2 } }],
    { x: 9.3, y: y + 0.14, w: 3.3, h: 1.15, fontSize: 12, valign: "middle" });
};
const fmt = (v, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const da = D.pairs["DA0.1-PCA20-Linear"];
if (da) stat(1.45, HEX.tintOrange, `${fmt(da.test.diff_avg)} pp`, "Decision-aware PCA (ours)",
  `PCA20-Linear: ${da.two_stage.avg.toFixed(1)}% → ${da.e2e.avg.toFixed(1)}% a year when the loadings learn from the portfolio (t = ${da.test.t.toFixed(1)})`, "stat-da");
const cr = D.costs && D.costs.rows;
if (cr) {
  const e = cr["E2E-tc20"], top = cr["MSE-Top"], tuned = cr["MSE-Opt (tuned)"];
  stat(2.97, HEX.tintBlue, `${e.to.toFixed(0)}%`, "Costs, 20 bps: daily turnover of E2E",
    `vs ${top.to.toFixed(0)}% for the MSE top decile. Net return ${e.avg.toFixed(1)}% vs ${top.avg.toFixed(1)}% (MSE-Top) and ${tuned.avg.toFixed(1)}% (tuned MSE-Opt, t = ${D.costs.tests["MSE-Opt (tuned)"].t.toFixed(1)})`, "stat-costs");
}
const mr = D.meansd && D.meansd.rows;
if (mr) {
  const e = mr["E2E mean-SD"], b = mr["MSE-Opt"], ppp = mr["PPP"];
  stat(4.49, HEX.tintTeal, e.sr.toFixed(2), "Mean–SD, η = 1: Sharpe of E2E",
    `vs ${b.sr.toFixed(2)} for MSE-Opt` + (ppp ? ` and ${ppp.sr.toFixed(2)} for PPP` : "") + `; volatility ${e.sd.toFixed(1)}% vs ${b.sd.toFixed(1)}% (difference t = ${D.meansd.tests["MSE-Opt"].t.toFixed(1)})`, "stat-meansd");
}
const by = (key) => Object.keys(S.models).filter((m) => !m.startsWith("E2E-") && !m.startsWith("DA"))
  .reduce((a, b) => (S.models[a][key] >= S.models[b][key] ? a : b));
const bestIC = by("IC"), bestSR = by("H_sr");
box(s, 0.6, 6.05, 12.15, 0.72, HEX.tintTeal, "finding-box", 0.06);
text(s, [
  { text: "Accuracy ≠ decision quality: ", options: { bold: true, color: C.text2 } },
  { text: `among two-stage models ${bestIC} has the best rank IC (${S.models[bestIC].IC.toFixed(3)}) but ${bestSR} the best long-only Sharpe (${S.models[bestSR].H_sr.toFixed(2)}).`, options: { color: C.text1 } },
], { x: 0.85, y: 6.05, w: 11.7, h: 0.72, valign: "middle", fontSize: 13 });
s.addNotes("Each E2E model is compared with the two-stage model it starts from, on the same test days, before costs. Differences are tested with Newey-West errors. Costs: NN3 trained with 20 bps costs against the MSE top decile and the validation-tuned MSE-Opt. Mean-SD: eta = 1, before costs.");

pres.writeFile({ fileName: OUT }).then(async () => {
  if (applyTheme) await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
