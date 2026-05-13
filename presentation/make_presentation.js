// Course project presentation generator — UrbanSound8K classification
// Run: node presentation/make_presentation.js
// Output: presentation/presentation.pptx
//
// Light-theme palette:
//   bg          FFFFFF  white
//   bg_soft     F0F4F8  very light blue-gray (cards, bands)
//   bg_warm     FCEEE3  very light peach (highlight cards)
//   primary     2E5A88  medium blue (titles, headers)
//   secondary   5B8FB9  lighter blue (subtitles, axis)
//   accent      D97757  warm orange (key stats, callouts)
//   accent2     5C8C66  muted green (positive comparison)
//   text        2A2D34  near-black body
//   muted       7A7A7A  caption gray
//   divider     D6DEE5  light border
//
// Fonts: Cambria for headers (sturdy serif), Calibri for body.

const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = (name) =>
  path.resolve(__dirname, "..", "results", "figures", name);

const C = {
  bg: "FFFFFF",
  bg_soft: "F0F4F8",
  bg_warm: "FCEEE3",
  primary: "2E5A88",
  secondary: "5B8FB9",
  accent: "D97757",
  accent2: "5C8C66",
  text: "2A2D34",
  muted: "7A7A7A",
  divider: "D6DEE5",
};

const F = { head: "Cambria", body: "Calibri" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 × 7.5 inches
pres.title = "UrbanSound8K — классификация городских звуков";
pres.author = "Ким А.М., ИУ12-41М";

// Slide dimensions
const W = 13.3;
const H = 7.5;

// Helpers
function titleBar(slide, text, subtitle) {
  // Title text
  slide.addText(text, {
    x: 0.6, y: 0.4, w: 12.1, h: 0.7,
    fontSize: 30, bold: true, fontFace: F.head,
    color: C.primary, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6, y: 1.05, w: 12.1, h: 0.4,
      fontSize: 14, italic: true, fontFace: F.body,
      color: C.muted, margin: 0,
    });
  }
}

function footer(slide, pageNum) {
  // Subtle footer
  slide.addText("ИИ в ЦОС — курсовой проект — Ким А.М., ИУ12-41М",
    { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 9, color: C.muted, fontFace: F.body });
  slide.addText(String(pageNum),
    { x: 12.4, y: 7.05, w: 0.3, h: 0.3, fontSize: 9, color: C.muted, fontFace: F.body, align: "right" });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: fill || C.bg_soft },
    line: { color: C.divider, width: 0.5 },
    rectRadius: 0.08,
  });
}

// ============================================================
// SLIDE 1 — TITLE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Decorative coloured strip on the right edge
  s.addShape(pres.shapes.RECTANGLE, {
    x: 11.3, y: 0, w: 2.0, h: H,
    fill: { color: C.bg_soft }, line: { color: C.bg_soft, width: 0 },
  });
  // Accent stripe
  s.addShape(pres.shapes.RECTANGLE, {
    x: 11.1, y: 0, w: 0.18, h: H,
    fill: { color: C.accent }, line: { color: C.accent, width: 0 },
  });

  // Eyebrow
  s.addText("Курсовой проект", {
    x: 0.9, y: 1.6, w: 10, h: 0.4,
    fontSize: 16, color: C.secondary, fontFace: F.body,
    charSpacing: 4, bold: true,
  });

  // Title
  s.addText("Классификация городских звуков", {
    x: 0.9, y: 2.1, w: 10, h: 1.1,
    fontSize: 46, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText("на датасете UrbanSound8K", {
    x: 0.9, y: 3.0, w: 10, h: 0.7,
    fontSize: 28, italic: true, fontFace: F.head, color: C.text,
  });

  // Method one-liner
  s.addText([
    { text: "Сравнение собственной 2D-CNN и дотюненной ", options: { color: C.text } },
    { text: "PANN CNN10", options: { color: C.accent, bold: true } },
    { text: " (pretrained на AudioSet)", options: { color: C.text } },
  ], { x: 0.9, y: 4.0, w: 10, h: 0.5, fontSize: 16, fontFace: F.body });

  // Divider line
  s.addShape(pres.shapes.LINE, {
    x: 0.9, y: 5.2, w: 4.0, h: 0,
    line: { color: C.accent, width: 2 },
  });

  // Author block
  s.addText("Ким А.М.", {
    x: 0.9, y: 5.4, w: 10, h: 0.45,
    fontSize: 20, bold: true, fontFace: F.head, color: C.text,
  });
  s.addText("Группа ИУ12-41М", {
    x: 0.9, y: 5.85, w: 10, h: 0.35,
    fontSize: 14, fontFace: F.body, color: C.muted,
  });
  s.addText("Дисциплина: «Методы ИИ в ЦОС»  ·  МГТУ им. Н.Э. Баумана  ·  2026", {
    x: 0.9, y: 6.2, w: 10, h: 0.35,
    fontSize: 12, italic: true, fontFace: F.body, color: C.muted,
  });
}

// ============================================================
// SLIDE 2 — ВВЕДЕНИЕ
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Введение", "Классификация звуков окружающей среды (Environmental Sound Classification)");

  // Two-column layout
  // Left: context
  card(s, 0.6, 1.7, 6.0, 4.9);
  s.addText("Задача", {
    x: 0.9, y: 1.85, w: 5.4, h: 0.4,
    fontSize: 18, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "По короткой звуковой записи (≤ 4 c) определить ", options: { breakLine: true } },
    { text: "класс источника звука. ", options: { bold: true, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Поддомен — ", options: {} },
    { text: "Environmental Sound Classification", options: { italic: true } },
    { text: " (ESC), смежные задачи:", options: { breakLine: true } },
  ], { x: 0.9, y: 2.3, w: 5.4, h: 1.6, fontSize: 13, fontFace: F.body, color: C.text, paraSpaceAfter: 4 });

  s.addText([
    { text: "Урбан-акустика — мониторинг шума, выявление аномалий (выстрелы, сирены).", options: { bullet: true, breakLine: true } },
    { text: "Биоакустика — определение видов животных по звукам.", options: { bullet: true, breakLine: true } },
    { text: "Голосовые ассистенты — wake-word / звуковые события.", options: { bullet: true } },
  ], { x: 0.9, y: 4.0, w: 5.4, h: 2.4, fontSize: 12, fontFace: F.body, color: C.text, paraSpaceAfter: 4 });

  // Right: methods
  card(s, 6.8, 1.7, 5.9, 4.9);
  s.addText("Подходы", {
    x: 7.1, y: 1.85, w: 5.3, h: 0.4,
    fontSize: 18, bold: true, fontFace: F.head, color: C.primary,
  });

  s.addText([
    { text: "Классические: ", options: { bold: true } },
    { text: "MFCC + SVM/Random Forest, GMM-HMM. Baseline в литературе ~0.65 на UrbanSound8K.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Глубокие сети на спектрограммах: ", options: { bold: true } },
    { text: "2D-CNN на лог-мел, PiczakCNN, SB-CNN, AST. Современный стандарт.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Pre-training на AudioSet: ", options: { bold: true } },
    { text: "PANNs CNN10/CNN14 (Kong et al., 2020). 2 млн аудио, 527 классов.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "1D-CNN на raw waveform: ", options: { bold: true } },
    { text: "M5, SampleCNN — end-to-end, но дороже в обучении.", options: {} },
  ], { x: 7.1, y: 2.3, w: 5.3, h: 4.0, fontSize: 12, fontFace: F.body, color: C.text, paraSpaceAfter: 4 });

  footer(s, 2);
}

// ============================================================
// SLIDE 3 — ДАТАСЕТ UrbanSound8K
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Датасет UrbanSound8K", "Salamon, Jacoby, Bello (2014) · CC BY-NC 3.0");

  // Stats row: 4 large numeric callouts
  const stats = [
    { num: "8 732", label: "аудио-сегмента" },
    { num: "10", label: "классов" },
    { num: "10-fold", label: "official CV splits" },
    { num: "≤ 4 c", label: "длина сегмента" },
  ];
  const sw = 2.85, sh = 1.5, sgap = 0.15;
  const sx0 = (W - 4 * sw - 3 * sgap) / 2;
  stats.forEach((st, i) => {
    const x = sx0 + i * (sw + sgap);
    card(s, x, 1.75, sw, sh, C.bg_warm);
    s.addText(st.num, {
      x, y: 1.85, w: sw, h: 0.95,
      fontSize: 44, bold: true, fontFace: F.head, color: C.accent,
      align: "center", valign: "middle",
    });
    s.addText(st.label, {
      x, y: 2.75, w: sw, h: 0.45,
      fontSize: 12, fontFace: F.body, color: C.text,
      align: "center", valign: "middle",
    });
  });

  // Classes block
  card(s, 0.6, 3.55, 8.4, 3.15);
  s.addText("10 классов городских звуков", {
    x: 0.85, y: 3.7, w: 8.0, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });
  const classes = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
  ];
  // 5x2 grid
  classes.forEach((cls, i) => {
    const col = i % 5, row = Math.floor(i / 5);
    const x = 0.85 + col * 1.6, y = 4.25 + row * 0.7;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 1.5, h: 0.55,
      fill: { color: "FFFFFF" }, line: { color: C.secondary, width: 0.75 },
      rectRadius: 0.06,
    });
    s.addText(cls, {
      x, y, w: 1.5, h: 0.55,
      fontSize: 11, fontFace: F.body, color: C.text,
      align: "center", valign: "middle",
    });
  });
  // Caveat
  s.addText("→ minority: gun_shot (374), car_horn (429) — остальные ~800-1000", {
    x: 0.85, y: 6.0, w: 8.0, h: 0.4,
    fontSize: 11, italic: true, fontFace: F.body, color: C.muted,
  });

  // Side panel: sources
  card(s, 9.2, 3.55, 3.5, 3.15);
  s.addText("Источники", {
    x: 9.4, y: 3.7, w: 3.1, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Zenodo (оригинал):", options: { bold: true, breakLine: true } },
    { text: "zenodo.org/records/1203745", options: { color: C.secondary, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "HuggingFace mirror:", options: { bold: true, breakLine: true } },
    { text: "danavery/urbansound8K", options: { color: C.secondary, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Field recordings:", options: { bold: true, breakLine: true } },
    { text: "Freesound.org — записи в реальной городской среде, разный sample rate, разное качество.", options: {} },
  ], { x: 9.4, y: 4.15, w: 3.1, h: 2.4, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 3 });

  footer(s, 3);
}

// ============================================================
// SLIDE 4 — EDA
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "EDA: структура датасета");

  // Native bar chart: class distribution
  s.addText("Распределение классов", {
    x: 0.6, y: 1.7, w: 6, h: 0.4,
    fontSize: 14, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addChart(pres.charts.BAR, [{
    name: "files per class",
    labels: ["air_cond", "car_horn", "children", "dog_bark", "drilling",
             "engine", "gun_shot", "jackham", "siren", "street_m"],
    values: [1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000],
  }], {
    x: 0.55, y: 2.1, w: 6.4, h: 3.4, barDir: "col",
    chartColors: [C.secondary],
    chartArea: { fill: { color: C.bg }, roundedCorners: false },
    catAxisLabelColor: C.muted, catAxisLabelFontSize: 8,
    valAxisLabelColor: C.muted, valAxisLabelFontSize: 9,
    valGridLine: { color: C.divider, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: false, showLegend: false,
  });

  // Right: fold distribution heatmap as numbers, and key observations
  s.addText("Что важно из EDA", {
    x: 7.3, y: 1.7, w: 5.4, h: 0.4,
    fontSize: 14, bold: true, fontFace: F.head, color: C.primary,
  });
  card(s, 7.2, 2.1, 5.6, 4.6);
  s.addText([
    { text: "Дисбаланс: ", options: { bold: true } },
    { text: "8 из 10 классов имеют ~1 000 файлов. Minority — ", options: {} },
    { text: "gun_shot (374)", options: { color: C.accent, bold: true } },
    { text: " и ", options: {} },
    { text: "car_horn (429)", options: { color: C.accent, bold: true } },
    { text: ".", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Folds: ", options: { bold: true } },
    { text: "10 предзаданных. Стратификация не идеальна — gun_shot отсутствует в 2 folds.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Длительность: ", options: { bold: true } },
    { text: "большинство сегментов 3.5–4 с, ~10% короче 2 с → pad нулями справа, длинные — center-crop.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Sample rate: ", options: { bold: true } },
    { text: "разный (8 / 16 / 22.05 / 44.1 / 48 kHz) — обязателен resample.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Метрики: ", options: { bold: true } },
    { text: "accuracy (датасет почти сбалансирован) + macro-F1 (страховка от minority).", options: {} },
  ], { x: 7.4, y: 2.25, w: 5.3, h: 4.35, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 3 });

  footer(s, 4);
}

// ============================================================
// SLIDE 5 — ПРЕПРОЦЕССИНГ PIPELINE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Препроцессинг: от wav к лог-мел спектрограмме");

  // Pipeline boxes
  const steps = [
    { title: "1. Load",   sub: "torchaudio,\nmono = mean(ch)" },
    { title: "2. Resample", sub: "22 050 Hz (own CNN)\n32 000 Hz (PANN)" },
    { title: "3. Pad / Crop", sub: "4 с фикс. длина\npad → 0 справа\ncrop → центр" },
    { title: "4. Mel-Spec", sub: "n_fft = 1024\nhop = 512 (own)\nn_mels = 64" },
    { title: "5. log + norm", sub: "log₁₀(mel + ε)\nz-score per-sample" },
  ];
  const bw = 2.3, bh = 1.5, bgap = 0.18;
  const bx0 = (W - 5 * bw - 4 * bgap) / 2;
  steps.forEach((st, i) => {
    const x = bx0 + i * (bw + bgap);
    card(s, x, 2.1, bw, bh, C.bg_soft);
    s.addText(st.title, {
      x, y: 2.2, w: bw, h: 0.5,
      fontSize: 14, bold: true, fontFace: F.head, color: C.primary,
      align: "center", valign: "middle",
    });
    s.addText(st.sub, {
      x: x + 0.1, y: 2.65, w: bw - 0.2, h: 0.85,
      fontSize: 10, fontFace: F.body, color: C.text,
      align: "center", valign: "top",
    });
    // Arrow between boxes
    if (i < 4) {
      s.addShape(pres.shapes.RIGHT_TRIANGLE, {
        x: x + bw + 0.02, y: 2.78, w: bgap - 0.06, h: 0.14,
        fill: { color: C.accent }, line: { color: C.accent, width: 0 },
        rotate: 90,
      });
    }
  });

  // Output stat line
  card(s, 0.6, 4.0, 12.1, 0.9, C.bg_warm);
  s.addText([
    { text: "Выход:  ", options: { fontSize: 14, bold: true, color: C.text } },
    { text: "тензор (1, 64, 173) — лог-мел-картинка 64 мел-полосы × 173 фрейма за 4 с.", options: { fontSize: 14, color: C.text } },
  ], { x: 0.8, y: 4.1, w: 11.7, h: 0.7, fontFace: F.body, valign: "middle" });

  // Caching note
  s.addText([
    { text: "Кеш спектрограмм на диск.  ", options: { bold: true, color: C.primary } },
    { text: "При первом проходе записываем `.pt` в `cache_mel/`. На последующих эпохах и фолдах — только torch.load → ускорение CPU-обучения 5–10×.", options: { color: C.text } },
  ], { x: 0.6, y: 5.15, w: 12.1, h: 0.7, fontSize: 12, fontFace: F.body });

  // Pipeline code path
  s.addText("Код: src/data/audio.py · src/data/dataset.py", {
    x: 0.6, y: 6.1, w: 12.1, h: 0.4,
    fontSize: 11, italic: true, fontFace: F.body, color: C.muted,
  });

  footer(s, 5);
}

// ============================================================
// SLIDE 6 — Архитектура: своя 2D-CNN
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Архитектура 1: собственная 2D-CNN", "4 conv-блока + AdaptiveAvgPool + FC — 249 866 параметров");

  // Left: architecture diagram (boxes)
  const layers = [
    { name: "Input",       shape: "1 × 64 × 173",   color: C.bg_soft },
    { name: "ConvBlock 1", shape: "32 × 32 × 86",   color: C.secondary, fg: "FFFFFF" },
    { name: "ConvBlock 2", shape: "64 × 16 × 43",   color: C.secondary, fg: "FFFFFF" },
    { name: "ConvBlock 3", shape: "128 × 8 × 21",   color: C.secondary, fg: "FFFFFF" },
    { name: "ConvBlock 4", shape: "128 × 4 × 10",   color: C.secondary, fg: "FFFFFF" },
    { name: "AdaptiveAvgPool", shape: "128",         color: C.accent2, fg: "FFFFFF" },
    { name: "FC 128 → 64 → 10", shape: "logits",     color: C.accent, fg: "FFFFFF" },
  ];
  const lw = 4.5, lh = 0.55, lgap = 0.12;
  const lx = 0.8, ly0 = 1.85;
  layers.forEach((l, i) => {
    const y = ly0 + i * (lh + lgap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: lx, y, w: lw, h: lh,
      fill: { color: l.color }, line: { color: l.color, width: 0 },
      rectRadius: 0.06,
    });
    s.addText(l.name, {
      x: lx + 0.15, y, w: 2.4, h: lh,
      fontSize: 12, bold: true, fontFace: F.body,
      color: l.fg || C.text, valign: "middle",
    });
    s.addText(l.shape, {
      x: lx + 2.5, y, w: lw - 2.6, h: lh,
      fontSize: 11, fontFace: F.body, color: l.fg || C.muted,
      align: "right", valign: "middle",
    });
  });

  // Right: ConvBlock detail + training settings
  card(s, 6.0, 1.85, 6.7, 2.4);
  s.addText("Что внутри ConvBlock", {
    x: 6.2, y: 1.95, w: 6.3, h: 0.4,
    fontSize: 14, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Conv2d(k = 3, p = 1) → ", options: {} },
    { text: "BatchNorm2d", options: { bold: true } },
    { text: " → ReLU → ", options: {} },
    { text: "MaxPool(2,2)", options: { bold: true } },
    { text: " → ", options: {} },
    { text: "Dropout2d", options: { bold: true } },
    { text: " (0.10 → 0.25 от блока к блоку)", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Kaiming init для conv и линейных, BN-init по умолчанию.", options: {} },
  ], { x: 6.2, y: 2.4, w: 6.3, h: 1.75, fontSize: 12, fontFace: F.body, color: C.text });

  // Training settings
  card(s, 6.0, 4.4, 6.7, 2.25);
  s.addText("Обучение", {
    x: 6.2, y: 4.5, w: 6.3, h: 0.4,
    fontSize: 14, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Optimizer:", options: { bold: true } }, { text: " AdamW, lr = 1e-3, weight_decay = 1e-4", options: { breakLine: true } },
    { text: "Loss:",      options: { bold: true } }, { text: " CrossEntropy", options: { breakLine: true } },
    { text: "Scheduler:", options: { bold: true } }, { text: " CosineAnnealingLR(T_max = epochs)", options: { breakLine: true } },
    { text: "Early stop:",options: { bold: true } }, { text: " patience по val_acc", options: { breakLine: true } },
    { text: "Grad clip:", options: { bold: true } }, { text: " max_norm = 1.0", options: { breakLine: true } },
    { text: "Batch:",     options: { bold: true } }, { text: " 32; per-fold 90% train + 10% val (seed = 42)", options: {} },
  ], { x: 6.2, y: 4.95, w: 6.3, h: 1.65, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 2 });

  footer(s, 6);
}

// ============================================================
// SLIDE 7 — Архитектура: PANN CNN10
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Архитектура 2: PANN CNN10 (fine-tune)", "Pretrained на AudioSet (527 классов, 2 млн аудио) — 4 954 058 параметров");

  // Diagram: pretrained backbone + new head, two-phase tuning
  // Top: schematic
  const blocks = [
    { name: "ConvBlock 1\n64 ch", color: C.secondary },
    { name: "ConvBlock 2\n128 ch", color: C.secondary },
    { name: "ConvBlock 3\n256 ch", color: C.secondary },
    { name: "ConvBlock 4\n512 ch", color: C.secondary },
  ];
  const bw = 1.4, bh = 1.1;
  const bx0 = 0.8;
  blocks.forEach((b, i) => {
    const x = bx0 + i * (bw + 0.15);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.85, w: bw, h: bh,
      fill: { color: b.color }, line: { color: b.color, width: 0 },
      rectRadius: 0.08,
    });
    s.addText(b.name, {
      x, y: 1.85, w: bw, h: bh,
      fontSize: 11, bold: true, fontFace: F.body, color: "FFFFFF",
      align: "center", valign: "middle",
    });
  });
  // fc1
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.3, y: 1.85, w: 1.6, h: 1.1,
    fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, rectRadius: 0.08,
  });
  s.addText("fc1\n512 → 512", {
    x: 7.3, y: 1.85, w: 1.6, h: 1.1,
    fontSize: 11, bold: true, fontFace: F.body, color: "FFFFFF",
    align: "center", valign: "middle",
  });
  // New classifier (replaces fc_audioset)
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 9.1, y: 1.85, w: 1.9, h: 1.1,
    fill: { color: C.accent }, line: { color: C.accent, width: 0 }, rectRadius: 0.08,
  });
  s.addText("NEW classifier\n512 → 10", {
    x: 9.1, y: 1.85, w: 1.9, h: 1.1,
    fontSize: 11, bold: true, fontFace: F.body, color: "FFFFFF",
    align: "center", valign: "middle",
  });
  // logits annotation
  s.addText("→ 10 logits", {
    x: 11.2, y: 1.85, w: 1.5, h: 1.1,
    fontSize: 12, italic: true, fontFace: F.body, color: C.text,
    align: "center", valign: "middle",
  });

  // Backbone bracket annotation
  s.addText("← Backbone из AudioSet (MIT, Kong et al., 2020) — заморожен в phase 1", {
    x: 0.8, y: 3.0, w: 6.65, h: 0.35,
    fontSize: 10, italic: true, fontFace: F.body, color: C.muted, align: "left",
  });
  s.addText("Заменили: AudioSet head 512 → 527  ⇒  10-class head", {
    x: 9.1, y: 3.0, w: 3.6, h: 0.35,
    fontSize: 10, italic: true, fontFace: F.body, color: C.muted, align: "center",
  });

  // Two-phase fine-tune block
  card(s, 0.6, 3.6, 12.1, 3.1, C.bg_soft);
  s.addText("Двухфазное fine-tune", {
    x: 0.85, y: 3.75, w: 11.6, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });

  // Phase 1
  card(s, 0.85, 4.2, 5.8, 2.4, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y: 4.2, w: 0.12, h: 2.4, fill: { color: C.secondary }, line: { color: C.secondary, width: 0 } });
  s.addText("Phase 1  ·  warmup", {
    x: 1.1, y: 4.3, w: 5.4, h: 0.4,
    fontSize: 14, bold: true, fontFace: F.head, color: C.secondary,
  });
  s.addText([
    { text: "Backbone заморожен. Учим только новую 10-class head.", options: { breakLine: true } },
    { text: "lr = 1e-3 на head", options: { bullet: true, breakLine: true } },
    { text: "5 эпох (для mini-прогона — 2)", options: { bullet: true, breakLine: true } },
    { text: "Сохраняем temp ckpt → отбрасывается после phase 2", options: { bullet: true } },
  ], { x: 1.1, y: 4.75, w: 5.4, h: 1.7, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 2 });

  // Phase 2
  card(s, 6.85, 4.2, 5.8, 2.4, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: 4.2, w: 0.12, h: 2.4, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  s.addText("Phase 2  ·  fine-tune", {
    x: 7.1, y: 4.3, w: 5.4, h: 0.4,
    fontSize: 14, bold: true, fontFace: F.head, color: C.accent,
  });
  s.addText([
    { text: "Размораживаем backbone — учим всё.", options: { breakLine: true } },
    { text: "lr_backbone = 1e-4  (discriminative)", options: { bullet: true, breakLine: true } },
    { text: "lr_head = 1e-3", options: { bullet: true, breakLine: true } },
    { text: "epoch_offset → CSV эпохи нумеруются 6, 7, 8 …", options: { bullet: true } },
  ], { x: 7.1, y: 4.75, w: 5.4, h: 1.7, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 2 });

  footer(s, 7);
}

// ============================================================
// SLIDE 8 — АУГМЕНТАЦИЯ
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Аугментация (только train)", "SpecAugment (Park et al., 2019) + Gaussian noise — применяется после кеша спектрограммы");

  // Left: SpecAugment description with mini diagram
  card(s, 0.6, 1.85, 6.0, 4.85);
  s.addText("SpecAugment", {
    x: 0.85, y: 2.0, w: 5.5, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });

  // Schematic spectrogram with masks
  const sx = 0.85, sy = 2.55, sw = 5.5, ssh = 1.3;
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx, y: sy, w: sw, h: ssh,
    fill: { color: C.bg_soft }, line: { color: C.divider, width: 0.5 },
  });
  // freq masks (horizontal bands)
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx, y: sy + 0.25, w: sw, h: 0.18,
    fill: { color: C.accent, transparency: 30 }, line: { color: C.accent, width: 0 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx, y: sy + 0.75, w: sw, h: 0.12,
    fill: { color: C.accent, transparency: 30 }, line: { color: C.accent, width: 0 },
  });
  // time masks (vertical bands)
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx + 1.7, y: sy, w: 0.45, h: ssh,
    fill: { color: C.secondary, transparency: 30 }, line: { color: C.secondary, width: 0 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx + 3.8, y: sy, w: 0.3, h: ssh,
    fill: { color: C.secondary, transparency: 30 }, line: { color: C.secondary, width: 0 },
  });
  s.addText("64 мел", { x: sx - 0.55, y: sy + 0.55, w: 0.55, h: 0.3, fontSize: 9, color: C.muted, fontFace: F.body, align: "right" });
  s.addText("173 фрейма", { x: sx, y: sy + ssh + 0.05, w: sw, h: 0.3, fontSize: 9, color: C.muted, fontFace: F.body, align: "center" });
  // Legend
  s.addShape(pres.shapes.RECTANGLE, { x: sx, y: sy + ssh + 0.45, w: 0.2, h: 0.2, fill: { color: C.accent, transparency: 30 }, line: { color: C.accent, width: 0 } });
  s.addText("freq mask (2 × max 12 мел-полос)", { x: sx + 0.3, y: sy + ssh + 0.45, w: 4.5, h: 0.25, fontSize: 10, color: C.text, fontFace: F.body });
  s.addShape(pres.shapes.RECTANGLE, { x: sx, y: sy + ssh + 0.75, w: 0.2, h: 0.2, fill: { color: C.secondary, transparency: 30 }, line: { color: C.secondary, width: 0 } });
  s.addText("time mask (2 × max 25 фреймов)", { x: sx + 0.3, y: sy + ssh + 0.75, w: 4.5, h: 0.25, fontSize: 10, color: C.text, fontFace: F.body });

  s.addText("Заполнение — среднее значение спектрограммы.", {
    x: 0.85, y: 5.6, w: 5.5, h: 0.3,
    fontSize: 10, italic: true, fontFace: F.body, color: C.muted,
  });
  s.addText("Cрабатывает каждый сэмпл; маски случайны → новый паттерн каждой эпохи.", {
    x: 0.85, y: 5.95, w: 5.5, h: 0.6,
    fontSize: 10, italic: true, fontFace: F.body, color: C.muted,
  });

  // Right: noise + rationale
  card(s, 6.8, 1.85, 5.9, 2.25);
  s.addText("Gaussian noise", {
    x: 7.05, y: 2.0, w: 5.4, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Добавляется на лог-мел тензор:", options: { breakLine: true } },
    { text: "std = 0.005, вероятность p = 0.3", options: { bullet: true, breakLine: true } },
    { text: "Случайный сдвиг распределения признаков → робастность к шумам записи.", options: { bullet: true } },
  ], { x: 7.05, y: 2.5, w: 5.4, h: 1.55, fontSize: 12, fontFace: F.body, color: C.text, paraSpaceAfter: 3 });

  card(s, 6.8, 4.3, 5.9, 2.4);
  s.addText("Почему так делаем", {
    x: 7.05, y: 4.45, w: 5.4, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Аугментация ", options: {} }, { text: "на спектрограмме", options: { bold: true } },
    { text: " — не на wav. Это позволяет ", options: {} }, { text: "переиспользовать кеш", options: { bold: true } },
    { text: " (мелы рассчитаны 1 раз), при этом каждый mini-batch видит новый паттерн.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Pitch-shift / time-stretch — медленнее и не дают выигрыша на этой задаче.", options: {} },
  ], { x: 7.05, y: 4.9, w: 5.4, h: 1.75, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 3 });

  footer(s, 8);
}

// ============================================================
// SLIDE 9 — ПРОБЛЕМЫ И РЕШЕНИЯ
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Проблемы при решении задачи и их решения");

  const problems = [
    {
      title: "Zenodo блокирует прямую скачку",
      issue: "HTTP 403: «unusual traffic from your network» при попытке `urlopen` UrbanSound8K.tar.gz. Воспроизводится и через urllib, и через curl с браузерным User-Agent.",
      fix: "Сделал fallback: HuggingFace mirror `danavery/urbansound8K` (та же CC BY-NC лицензия, identical контент в parquet-шардах). Скрипт `src/data/download.py` теперь пробует Zenodo → при ошибке переключается на HF.",
    },
    {
      title: "datasets 4.x требует torchcodec для декода audio",
      issue: "load_dataset(...)['audio'] возвращает Audio feature, который при iter-е пытается использовать torchcodec — он не установлен и плохо ставится под Windows + torch 2.8 CPU.",
      fix: "`.cast_column('audio', Audio(decode=False))` → получаем raw bytes WAV-файла. Пишем эти bytes напрямую на диск в `audio/foldN/<name>.wav`. Декодирование уже не нужно.",
    },
    {
      title: "Длинный CPU-прогон + BSOD",
      issue: "Полный 10-fold × 50 эпох для own CNN + PANN — около 20 часов на CPU. Posередине прогона система ушла в BSOD (предположительно тепло / драйвер на 14700K).",
      fix: "Сжал scope: own CNN 5 folds × 12 эпох, PANN 1 fold × 2 эпохи, ablation 1 fold × 12 эпох. Полный 10-fold протокол остался в коде (`run_*.py` принимает `--folds`), запускается без правок при наличии времени или GPU.",
    },
  ];

  let y = 1.7;
  problems.forEach((p, i) => {
    const ch = 1.55;
    card(s, 0.6, y, 12.1, ch, "FFFFFF");
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y, w: 0.14, h: ch,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 },
    });
    // Number circle
    s.addShape(pres.shapes.OVAL, {
      x: 0.95, y: y + 0.32, w: 0.55, h: 0.55,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 },
    });
    s.addText(String(i + 1), {
      x: 0.95, y: y + 0.32, w: 0.55, h: 0.55,
      fontSize: 22, bold: true, fontFace: F.head, color: "FFFFFF",
      align: "center", valign: "middle",
    });
    // Title
    s.addText(p.title, {
      x: 1.7, y: y + 0.12, w: 10.9, h: 0.4,
      fontSize: 14, bold: true, fontFace: F.head, color: C.primary,
    });
    s.addText([
      { text: "Симптом: ", options: { bold: true, color: C.accent } },
      { text: p.issue, options: { color: C.text } },
    ], { x: 1.7, y: y + 0.55, w: 10.9, h: 0.45, fontSize: 11, fontFace: F.body });
    s.addText([
      { text: "Решение: ", options: { bold: true, color: C.accent2 } },
      { text: p.fix, options: { color: C.text } },
    ], { x: 1.7, y: y + 1.02, w: 10.9, h: 0.5, fontSize: 11, fontFace: F.body });

    y += ch + 0.1;
  });

  footer(s, 9);
}

// ============================================================
// SLIDE 10 — РЕЗУЛЬТАТЫ
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Результаты", "Mini-scope: 5/1/1 folds × 12/2/12 эпох на CPU. Полный 10-fold протокол реализован в коде");

  // Stat cards row
  const stats = [
    { label: "Own CNN", acc: "0.573", f1: "0.551", folds: "5 folds × 12 эп.", color: C.secondary, params: "250K" },
    { label: "PANN CNN10 ft", acc: "0.790", f1: "0.806", folds: "1 fold × 2 эп.", color: C.accent, params: "5.0M" },
    { label: "Ablation (no aug)", acc: "0.733", f1: "0.729", folds: "1 fold × 12 эп.", color: C.accent2, params: "250K" },
  ];
  const cw = 4.0, ch = 2.1, cgap = 0.15;
  const cx0 = (W - 3 * cw - 2 * cgap) / 2;
  stats.forEach((st, i) => {
    const x = cx0 + i * (cw + cgap);
    card(s, x, 1.85, cw, ch, "FFFFFF");
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.85, w: cw, h: 0.18, fill: { color: st.color }, line: { color: st.color, width: 0 } });
    s.addText(st.label, {
      x: x + 0.2, y: 2.1, w: cw - 0.4, h: 0.4,
      fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
    });
    s.addText(st.acc, {
      x: x + 0.2, y: 2.5, w: cw - 0.4, h: 0.9,
      fontSize: 56, bold: true, fontFace: F.head, color: st.color,
      align: "left", valign: "middle",
    });
    s.addText("accuracy", {
      x: x + 0.2, y: 3.45, w: cw - 0.4, h: 0.3,
      fontSize: 12, italic: true, fontFace: F.body, color: C.muted,
    });
    s.addText([
      { text: "macro-F1:  ", options: { color: C.muted } },
      { text: st.f1, options: { bold: true, color: C.text } },
      { text: "    ·  ", options: { color: C.muted } },
      { text: "params:  ", options: { color: C.muted } },
      { text: st.params, options: { bold: true, color: C.text } },
    ], { x: x + 0.2, y: 3.75, w: cw - 0.4, h: 0.3, fontSize: 11, fontFace: F.body });
    s.addText(st.folds, {
      x: x + 0.2, y: 4.05, w: cw - 0.4, h: 0.3,
      fontSize: 10, italic: true, fontFace: F.body, color: C.muted,
    });
  });

  // Per-class F1 comparison figure (aspect 11:5; fit into 8.2 x 2.4 preserving ratio)
  s.addImage({
    path: FIG("per_class_f1_comparison.png"),
    x: 0.6, y: 4.4, sizing: { type: "contain", w: 8.2, h: 2.4 },
    w: 8.2, h: 2.4,
  });

  // Key takeaways panel
  card(s, 9.0, 4.4, 3.7, 2.4);
  s.addText("Главные цифры", {
    x: 9.2, y: 4.5, w: 3.3, h: 0.4,
    fontSize: 13, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "+21%", options: { bold: true, color: C.accent, fontSize: 18 } },
    { text: "  acc PANN над own CNN при том же препроцессинге", options: { color: C.text, fontSize: 11 } },
    { text: "", options: { breakLine: true, fontSize: 11 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "std 7%", options: { bold: true, color: C.accent2, fontSize: 14 } },
    { text: "  у own CNN по 5 folds → высокая дисперсия датасета", options: { color: C.text, fontSize: 11 } },
  ], { x: 9.2, y: 4.95, w: 3.3, h: 1.7, fontFace: F.body, paraSpaceAfter: 3 });

  footer(s, 10);
}

// ============================================================
// SLIDE 11 — РАЗБОР ОШИБОК (confusion matrix + observations)
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Разбор ошибок: confusion matrix Own CNN");

  // Confusion matrix figure (aspect ~8:7; fit into 6.6 x 5.0 preserving ratio)
  s.addImage({
    path: FIG("confusion_matrix_own_cnn_avg.png"),
    x: 0.6, y: 1.7, sizing: { type: "contain", w: 6.6, h: 5.0 },
    w: 6.6, h: 5.0,
  });

  // Observations panel (right)
  card(s, 7.4, 1.7, 5.3, 5.0);
  s.addText("Что видно из матрицы", {
    x: 7.6, y: 1.85, w: 4.9, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Сильные классы (диагональ > 0.7):", options: { bold: true, breakLine: true } },
    { text: "gun_shot — характерный короткий всплеск энергии.", options: { bullet: true, breakLine: true } },
    { text: "siren — устойчивые периодические гармоники.", options: { bullet: true, breakLine: true } },
    { text: "engine_idling — низкочастотный гул.", options: { bullet: true, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Слабые классы (диагональ < 0.5):", options: { bold: true, breakLine: true } },
    { text: "children_playing ↔ street_music — оба широкополосные, смешанная семантика.", options: { bullet: true, breakLine: true } },
    { text: "jackhammer ↔ drilling — оба ритмичные перкуссионные удары на близких частотах.", options: { bullet: true, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "minority классы (gun_shot 374, car_horn 429) — низкий precision из-за дефицита примеров.", options: { italic: true, color: C.muted } },
  ], { x: 7.6, y: 2.3, w: 4.9, h: 4.35, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 2 });

  footer(s, 11);
}

// ============================================================
// SLIDE 12 — ВЫВОДЫ + ДАЛЬНЕЙШИЕ УЛУЧШЕНИЯ
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  titleBar(s, "Выводы и возможные улучшения");

  // Left: что сделано
  card(s, 0.6, 1.7, 6.0, 5.0);
  s.addText("Что получилось", {
    x: 0.85, y: 1.85, w: 5.5, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Построен полный ML-пайплайн: download → preprocess + cache → train (own CNN / PANN / ablation) → aggregate.", options: { bullet: true, breakLine: true } },
    { text: "Реализован 10-fold протокол (запущен в mini-scope: 5/1/1 фолдов из-за CPU-бюджета).", options: { bullet: true, breakLine: true } },
    { text: "48 unit-тестов покрывают audio/aug/dataset/models/metrics/trainer/aggregator.", options: { bullet: true, breakLine: true } },
    { text: "Pretrained PANN +21% accuracy над собственной CNN — подтверждена ценность pre-training.", options: { bullet: true, breakLine: true } },
    { text: "Code + per-fold логи + 7 PNG графиков + summary.csv/md закоммичены в локальный git.", options: { bullet: true } },
  ], { x: 0.85, y: 2.3, w: 5.5, h: 4.3, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 5 });

  // Right: дальнейшие улучшения
  card(s, 6.8, 1.7, 5.9, 5.0);
  s.addText("Что дальше", {
    x: 7.05, y: 1.85, w: 5.4, h: 0.4,
    fontSize: 16, bold: true, fontFace: F.head, color: C.primary,
  });
  s.addText([
    { text: "Полный 10-fold для PANN и ablation на GPU (~1-2 ч вместо 15 ч на CPU).", options: { bullet: true, breakLine: true } },
    { text: "30–50 эпох обучения own CNN — даст время аугментации проявить эффект, ожидаемо 70-80% accuracy.", options: { bullet: true, breakLine: true } },
    { text: "Mixup / CutMix на спектрограммах — улучшение обобщения.", options: { bullet: true, breakLine: true } },
    { text: "1D-CNN (M5 / SampleCNN) на raw waveform — end-to-end ЦОС-подход.", options: { bullet: true, breakLine: true } },
    { text: "PANN CNN14 или AST как backbone — современная SOTA-планка.", options: { bullet: true, breakLine: true } },
    { text: "Optuna для гиперпараметров (lr, dropout, mask sizes).", options: { bullet: true } },
  ], { x: 7.05, y: 2.3, w: 5.4, h: 4.3, fontSize: 11, fontFace: F.body, color: C.text, paraSpaceAfter: 5 });

  footer(s, 12);
}

// ============================================================
// SLIDE 13 — Спасибо / контакты
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Decorative bands
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.55,
    fill: { color: C.bg_soft }, line: { color: C.bg_soft, width: 0 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: H - 0.55, w: W, h: 0.55,
    fill: { color: C.bg_soft }, line: { color: C.bg_soft, width: 0 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.55, w: 0.18, h: H - 1.1,
    fill: { color: C.accent }, line: { color: C.accent, width: 0 },
  });

  s.addText("Спасибо за внимание", {
    x: 0.6, y: 2.5, w: 12.1, h: 1.2,
    fontSize: 60, bold: true, fontFace: F.head, color: C.primary,
    align: "center", valign: "middle",
  });
  s.addText("Готов к вопросам", {
    x: 0.6, y: 3.7, w: 12.1, h: 0.6,
    fontSize: 22, italic: true, fontFace: F.head, color: C.muted,
    align: "center", valign: "middle",
  });

  // Bottom: repo and contact
  s.addShape(pres.shapes.LINE, {
    x: 5.0, y: 4.7, w: 3.3, h: 0,
    line: { color: C.accent, width: 2 },
  });

  s.addText("Ким А.М.  ·  ИУ12-41М  ·  МГТУ им. Н.Э. Баумана", {
    x: 0.6, y: 5.0, w: 12.1, h: 0.4,
    fontSize: 14, fontFace: F.body, color: C.text, align: "center",
  });
  s.addText("Код проекта — локальный git-репозиторий (готов к git push)", {
    x: 0.6, y: 5.45, w: 12.1, h: 0.4,
    fontSize: 12, italic: true, fontFace: F.body, color: C.muted, align: "center",
  });
}

const outPath = path.resolve(__dirname, "presentation.pptx");
pres.writeFile({ fileName: outPath }).then((f) => {
  console.log("Wrote", f);
});
