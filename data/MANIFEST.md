# Corpus Manifest

**Corpus name:** CyberWell Online Antisemitism Research Reports

**Domain:** Online antisemitism monitoring, platform policy, and digital safety research

**Source of documents:** Public PDF reports published by [CyberWell](https://cyberwell.org/) in `data/raw/`

**Number of documents:** 25 PDF files

**Approximate size** (measured from `data/processed/` after `python src/build_index.py`):

| Metric | Value |
|--------|--------|
| Total PDF pages | ~489 |
| Approximate corpus words | ~104,000 |
| Approximate tokens | ~135,000 |
| Chunks (paragraph_window, 2000w / 200 overlap) | 70 |

**File types:** PDF

**License / permission:** PDFs are public research reports published on [CyberWell](https://cyberwell.org/), used for a **course assignment**. This repository submits code, manifest, and evaluation artifacts; full PDFs may be omitted from git if not redistributed—obtain originals from CyberWell’s site. Not licensed for commercial reuse of report content.

**Why this corpus is suitable for RAG:**
- Contains domain-specific findings, statistics, case studies, and policy recommendations that a general-purpose LLM may not know accurately or may conflate
- Reports reference specific events, platforms, time periods, and organizational recommendations
- Retrieval grounds answers in cited source material rather than parametric knowledge

**What kind of questions should the system answer:**
- **Factual:** What did CyberWell report about antisemitism trends on a given platform?
- **Numerical:** What volumes, percentages, or counts are cited in a report?
- **Temporal:** What happened during a specific period (e.g., elections 2024–2025)?
- **Comparison:** How do findings differ across platforms or report editions?
- **Negation / absence:** What does the corpus *not* claim about a topic?

## Files in `data/raw/`

| File | Topic (approximate) |
|------|---------------------|
| `3.2.2026-Bondi-Beach-Report_CyberWell.pdf` | Bondi Beach incident analysis |
| `AI-Generated-Antisemitism_CyberWell-12.5.2026.pdf` | AI-generated antisemitic content |
| `Annual-Report-_-The-State-of-Online-Antisemitism-in-2022.pdf` | 2022 annual state of online antisemitism |
| `Antisemitism-Online-Amid-National-Elections-2024-2025.pdf` | Antisemitism amid national elections (2024–2025) |
| `Antisemitism-Trend-Alert-_-Denial-of-the-October-7-Massacre-on-Social-Media-Platforms-1.pdf` | October 7 massacre denial trend alert |
| `CyberWell-Alert-_-Online-Antisemitism-Spikes-in-Response-to-Ye.pdf` | Antisemitism spike in response to Ye (Kanye) |
| `CyberWell-Annual-Report-2025.pdf` | 2025 annual overview |
| `CyberWell-Policy-Recommendations-_-Regarding-Meta-Oversight-Board-Cases-Involving-Symbols-Adopted-by-Dangerous-Organizations.pdf` | Meta Oversight Board — symbols adopted by dangerous organizations |
| `CyberWell-Policy-Recommendations-_-Regarding-Metas-Policy-Advisory-Opinion-Request_-Shaheed-and-Designated-Dangerous-Individuals.pdf` | Meta PAO — Shaheed and designated dangerous individuals |
| `CyberWell-Policy-Recommendations-_-Regarding-Metas-Policy-Advisory-Opinion-Request-on_-2.pdf` | Meta policy advisory opinion (additional request) |
| `CyberWell-Policy-Recommendations-_-Regarding-Metas-Policy-Advisory-Opinion-Request-on-Holocaust-Denial-1.pdf` | Meta PAO — Holocaust denial |
| `Data-Insights-_-The-State-of-Antisemitism-on-Twitter.pdf` | Antisemitism on Twitter (data insights) |
| `Denial-Conspiratorial-Self-Victimization-Report-Public-Version.pdf` | Denial, conspiratorial, and self-victimization narratives |
| `Israel-Hamas-War-November-2023-_-Trending-Antisemitic-Narratives-Calls-to-Violence-1.pdf` | Israel–Hamas war (Nov 2023) — narratives and calls to violence |
| `Monetized-Antisemitism-on-YouTube-1.pdf` | Monetized antisemitism on YouTube |
| `October-7-Denial-_-The-Online-Attempt-to-Erase-Mass-Sexual-Violence-Against-Israeli-Women.pdf` | October 7 denial — sexual violence against Israeli women |
| `Online-Antisemitic-Election-Narratives-_-2024-U.S.-Elections.pdf` | 2024 U.S. election antisemitic narratives |
| `Online-Antisemitism-2023-_-Annual-Report.pdf` | 2023 annual report |
| `Online-Antisemitism-2024-_-Annual-Report-1.pdf` | 2024 annual report |
| `Report-_-Holocaust-Denial-and-Distortion-on-Social-Media-_-Yom-HaShoah-2023.pdf` | Holocaust denial and distortion (Yom HaShoah 2023) |
| `Report-_-The-Judeo-Masonic-Conspiracy-Theory-on-Meta.pdf` | Judeo-Masonic conspiracy theory on Meta |
| `The-Evolution-of-Online-Antisemitism-_-Pre-Post-October-7.pdf` | Evolution of online antisemitism pre/post October 7 |
| `The-Evolution-of-Online-Antisemitism-Report-for-the-Royal-Commission-.pdf` | Royal Commission submission |
| `Yes-Online-Antisemitic-Rants-_-2025-vs.-2022.pdf` | Ye antisemitic rants — 2025 vs. 2022 |
| `Yom-HaShoah-2024-_-Online-Holocaust-Hate-Speech-Narratives-and-Trends_.pdf` | Holocaust hate speech narratives (Yom HaShoah 2024) |
