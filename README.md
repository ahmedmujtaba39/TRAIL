# TRAIL — Tracking Resource Availability In (sign) Languages

A survey of ~50 national/regional sign languages assessing candidacy for **zero-shot continuous sign language recognition (CSLR)** research: languages where a public **isolated-word dictionary/lexicon** exists but **no substantial public continuous (sentence-level, natural-signing) video corpus** could be found.

## Methodology & important caveats

- Findings come from web searches across academic sources (ACL Anthology, arXiv, IEEE Xplore, ScienceDirect, PMC, LREC/FG proceedings), dataset repositories (Zenodo, Kaggle, Mendeley, HuggingFace, Papers with Code), and dictionary/lexicon projects (Spread The Sign, national deaf-association dictionaries, university lexicon archives), done by six parallel research passes (one per region below) in **2026**.
- **"No continuous corpus found" is an absence-of-evidence claim, not proof of absence.** A corpus may exist in a paper, institutional archive, or language behind a paywall/regional site that search did not surface. Treat every "Excellent" verdict below as a **starting hypothesis to verify manually** (contact national deaf federations, check institutional repositories) before committing research effort or funding to it.
- Ratings are relative, not absolute: ⭐⭐⭐ Excellent = solid isolated lexicon + no continuous corpus found; ⭐⭐ Good = same but with a thinner/less-verified isolated lexicon; ⭐ Weak = isolated resource itself is unverified/uncertain; **Not a candidate** = a continuous corpus was found, disqualifying it.
- Several languages initially assumed to be "safe" zero-shot targets turned out to already have continuous corpora once checked (Turkish, Azerbaijani, Japanese, Korean, Hong Kong, Ukrainian, Peruvian, Argentine, Nigerian, Ghanaian, Kenyan, South African, Ethiopian, Bangladeshi, Malaysian) — don't assume, verify per-language.
- Sources are cited inline per language in the regional tables below; every claim should be independently re-checked before use in a publication or grant application.

## Master ranked list — strongest zero-shot candidates (⭐⭐⭐ Excellent)

| Language | Region | Isolated resource | Confidence |
|---|---|---|---|
| Jordanian Sign Language | Middle East | 500-sign dictionary (2006) + Holy Land Institute for the Deaf dictionary | High |
| Palestinian Sign Language | Middle East | Print dictionaries (1992, 2014 rev.) ~2,000 signs | High |
| Yemeni Sign Language | Middle East | ArYSL dataset from Unified Yemeni Dictionary, 357 words (public, figshare) | High |
| Iranian/Persian Sign Language | Middle East | ISLR101 (arXiv, public) + Global Signbank archive | High |
| Thai Sign Language | SE Asia | TSL-ONE-S (4,152 videos/184 glosses, 2025, public) | High |
| Philippine Sign Language (FSL) | SE Asia | FSL-105 (2,130 videos/105 signs, public) | High |
| Algerian Sign Language | Africa | Alabib-65, ALGSL89, 3DZSignDB (multiple public datasets) | High |
| Moroccan Sign Language | Africa | MoSL, MSL datasets (public) | High |
| Serbian Sign Language | Europe | Confirmed on Spread The Sign | High |
| Croatian Sign Language | Europe | Confirmed on Spread The Sign + Hamburg SL Compendium | High |
| Romanian Sign Language | Europe | Spread The Sign + RoCoISLR (9,000+ videos, arXiv 2511.12767 — paper itself confirms no continuous set exists) | High |
| Bulgarian Sign Language | Europe | Spread The Sign + Bulgarian Deaf Union dictionaries | High |
| Mexican Sign Language (LSM) | Latin America | MX-ITESO-100, MSL-150, RGB-D corpus (multiple public datasets) | High |
| Taiwanese Sign Language | East Asia | TSL Online Dictionary (National Chung Cheng University, public) | Medium-High |
| Kuwaiti Sign Language | Middle East | KISR multi-dictionary hub (public) | Medium |
| Egyptian Sign Language | Africa | National Association of the Deaf (Cairo) dictionary + research sets | Medium |
| Nepalese Sign Language | S Asia | Official NSL Dictionary (4,300 signs), NSL23 | Medium |
| Sri Lankan Sign Language | S Asia | SSL400, SSL50 (Kaggle) | Medium |
| Vietnamese Sign Language | SE Asia | VSL400 (74,259 clips, public) | Medium |
| Mongolian Sign Language | East Asia | mnsl.mn official online dictionary (2024) | Medium |

**Secondary tier (⭐⭐ Good — plausible but thinner/less-verified isolated resources):** Tunisian, Iraqi, Lebanese, Pakistani, Indonesian (BISINDO), Albanian, Bosnian, Georgian, Chilean, Ecuadorian, Venezuelan.

**Weak/uncertain (⭐ — isolated resource itself not confirmed public):** Libyan, Uzbek, Colombian (has a small continuous corpus, CoL-SLTD, so weak rather than excellent).

**Disqualified — a continuous corpus already exists publicly:** Nigerian, Ghanaian, Kenyan, South African, Ethiopian (all via AfriSign and/or dedicated corpora), Turkish (E-TSL), Azerbaijani (AzSLD_Sentences), Bangladeshi (BTVSL), Malaysian (BIM-SSD), Japanese (NII JSL Colloquial Corpus), Korean (KSL-Guide, KETI), Hong Kong (TVB-HKSL-News), Ukrainian (UkrSL), Peruvian (PeruSIL/AEC), Argentine (LSA-T), Armenian (disqualified for a different reason — no usable public isolated lexicon was found either).

---

## Regional detail

### Africa

| Language | Isolated Dictionary Exists? | Continuous Corpus Found? | Verdict | Confidence |
|---|---|---|---|---|
| Egyptian Sign Language | Some — Cairo National Association of the Deaf online dictionary, 1984 print dictionary, small research sets ([Wikipedia](https://en.wikipedia.org/wiki/Egyptian_Sign_Language), [arXiv:2107.13647](https://arxiv.org/pdf/2107.13647)) | None found — 2025 review calls it "understudied" ([SciTePress ICAART 2025](https://www.scitepress.org/Papers/2025/133801/133801.pdf)) | ⭐⭐⭐ Excellent | Medium |
| Algerian Sign Language | Yes — Alabib-65 ([ACM TALLIP](https://dl.acm.org/doi/10.1145/3596909)), ALGSL89, 3DZSignDB | None found | ⭐⭐⭐ Excellent | High |
| Moroccan Sign Language | Yes — MoSL ([Mendeley](https://data.mendeley.com/datasets/23phgyt3mt/1)), MSL, IFES electoral lexicon | None found — papers list continuous work as future work | ⭐⭐⭐ Excellent | High |
| Tunisian Sign Language | Some — fragmented (TunSL-D, AVST medical dictionary, ATILS) | None found | ⭐⭐ Good | Medium |
| Libyan Sign Language | Unclear/Weak — no confirmed public dictionary | None found | ⭐ Weak | Low |
| Nigerian Sign Language | Yes — HuggingFace/Lanfrica set, S-DELI documentation project | Partial — AfriSign (Bible-verse sentence corpus) ([Springer](https://link.springer.com/article/10.1007/s44163-025-00227-7)) | Not a candidate | Medium |
| Ghanaian Sign Language | Yes — GSL lexicon, SignTalk-GSL | Yes — AfriSign + healthcare translation corpus ([Nature 2026](https://www.nature.com/articles/s41598-026-43478-9)) | Not a candidate | High |
| Kenyan Sign Language | Yes — HamNoSys vocabulary, 20k-video pose dataset | Yes — AI4KSL (~14,000 sentences, [arXiv:2410.18295](https://arxiv.org/abs/2410.18295)) + AfriSign | Not a candidate | High |
| South African Sign Language | Yes — UCT dataset, PanSALB/NID "DEAFinition" app | Yes — Wehrmeyer's SASL interpreting corpus + AfriSign | Not a candidate | High |
| Ethiopian Sign Language | Yes — AAU digital dictionary, 3,000+ signs | Yes — CESLR/CESL on Zenodo ([Zenodo](https://zenodo.org/records/10800699)) | Not a candidate | High |

### Middle East / Central Asia

| Language | Isolated Dictionary Exists? | Continuous Corpus Found? | Verdict | Confidence |
|---|---|---|---|---|
| Kuwaiti Sign Language | Yes — KISR multi-dictionary hub ([ssdd.kisr.edu.kw](https://ssdd.kisr.edu.kw/ada/sldictionary/index_en.html)) | None found (Arabic continuous sets found — ArabSign, Isharah, KArSL — are Saudi, not Kuwaiti) | ⭐⭐⭐ Excellent | Medium |
| Iraqi Sign Language | Some — dictionary-derived isolated ML set, public accessibility unclear | None found | ⭐⭐ Good | Medium |
| Jordanian Sign Language | Yes — 500-sign dictionary (2006) + Holy Land Institute for the Deaf dictionary ([LOT Publications](https://www.lotpublications.nl/Documents/193_fulltext.pdf)) | None found | ⭐⭐⭐ Excellent | High |
| Lebanese Sign Language | Some — informal curated video dictionary ([Sign Academy](https://signacademy.org/sign-language/lebanese-sign-language/)) | None found | ⭐⭐ Good | Medium |
| Palestinian Sign Language | Yes — print dictionaries (1992, 2014 rev., ~2,000 signs) | None found (only a small isolated math-gesture set) | ⭐⭐⭐ Excellent | High |
| Yemeni Sign Language | Yes — ArYSL dataset from Unified Yemeni Dictionary, 357 words, public ([figshare](https://figshare.com/articles/dataset/_b_Yemeni_sign_Language_dataset_b_/26114395)) | None found | ⭐⭐⭐ Excellent | High |
| Iranian/Persian Sign Language | Yes — ISLR101 ([arXiv:2503.12451](https://arxiv.org/abs/2503.12451)) + Global Signbank archive | None found | ⭐⭐⭐ Excellent | High |
| Turkish Sign Language (TİD) | Yes — AUTSL, BosphorusSign22k | **Yes** — E-TSL continuous corpus ([arXiv:2405.02984](https://arxiv.org/abs/2405.02984)) | **Not a candidate** | High |
| Azerbaijani Sign Language | Yes — AzSLD_Words | **Yes** — AzSLD_Sentences, 60+ hrs ([arXiv:2411.12865](https://arxiv.org/abs/2411.12865)) | **Not a candidate** | High |
| Uzbek Sign Language | Weak/uncertain — dictionary still "in development" as of 2025 ([Sharoit Plus](https://sharoitplus.uz/en/development-of-the-uzbek-sign-language-usl-through-the-creation-of-an-online-dictionary-website-and-mobile-application/)) | None found | ⭐ Weak | Low |

### South / Southeast Asia

| Language | Isolated Dictionary Exists? | Continuous Corpus Found? | Verdict | Confidence |
|---|---|---|---|---|
| Bangladeshi Sign Language | Yes — BdSL36, BdSLW60/102, BDSL49, KU-BdSL | **Yes** — BTVSL (60 hrs broadcast), BdSL-Continuous-1200 | **Not a candidate** | Medium |
| Pakistani Sign Language | Some — WLPSL (31 classes, Kaggle) | None found at sentence level (only continuous fingerspelling) | ⭐⭐ Good | Medium |
| Nepalese Sign Language | Yes — official NSL Dictionary (4,300 signs), NSL23 | Partial/not public — 40-sentence CSLR study, not released | ⭐⭐⭐ Excellent | Medium |
| Sri Lankan Sign Language | Yes — SSL400, SSL50 (Kaggle) | Partial/unclear public status | ⭐⭐⭐ Excellent | Medium |
| Indonesian Sign Language (BISINDO) | Yes — Word-Level BISINDO (Kaggle) | Partial, public-ish — DKI Jakarta sentence corpus (900→3,600 videos) | ⭐⭐ Good | Medium |
| Malaysian Sign Language (BIM) | Yes — BIM-SIGN, MyWSL | **Yes** — BIM-SSD v1/v2 (4,858–4,900 video/gloss/translation entries) | **Not a candidate** | Medium |
| Philippine Sign Language (FSL) | Yes — FSL-105 (2,130 videos, Mendeley/HuggingFace) | None found | ⭐⭐⭐ Excellent | High |
| Vietnamese Sign Language | Yes — VSL400 (74,259 clips, public) | Small/unclear public status | ⭐⭐⭐ Excellent | Medium |
| Thai Sign Language | Yes — TSL-ONE-S (4,152 videos, 2025, public) | None — 2026 paper states continuous TSL corpora are absent | ⭐⭐⭐ Excellent | High |

### East Asia

| Language | Isolated Dictionary Exists? | Continuous Corpus Found? | Verdict | Confidence |
|---|---|---|---|---|
| Japanese Sign Language | Yes — New JSL Dictionary, Asian Signbank | **Yes** — NII JSL Colloquial Corpus, 40 hrs ([research.nii.ac.jp](http://research.nii.ac.jp/jsl-corpus/)) | **Not a candidate** | High |
| Korean Sign Language | Yes — National Institute of Korean Language KSL Dictionary | **Yes** — KSL-Guide (121K samples), KETI dataset | **Not a candidate** | High |
| Taiwanese Sign Language | Yes — TSL Online Dictionary ([twtsl.ccu.edu.tw](https://twtsl.ccu.edu.tw/), ~1,000 items) | None found | ⭐⭐⭐ Excellent | Medium-High |
| Hong Kong Sign Language | Yes — HKSL Browser/Asian Signbank (CUHK, 3,700+ words) | **Yes** — TVB-HKSL-News, 16 hrs, 7,160 glosses ([LREC 2024](https://tvb-hksl-news.github.io/)) | **Not a candidate** | High |
| Mongolian Sign Language | Some — mnsl.mn official dictionary (2024) | None found | ⭐⭐⭐ Excellent | Medium |

### Europe

| Language | Isolated Dictionary Exists? | Continuous Corpus Found? | Verdict | Confidence |
|---|---|---|---|---|
| Albanian Sign Language | Some — small SignWriting-based dictionary, not on Spread The Sign | None found | ⭐⭐ Good | Medium |
| Bosnian Sign Language | Some — dictionaries tied to Yugoslav Sign Language | None found | ⭐⭐ Good | Low-Medium |
| Serbian Sign Language | Yes — confirmed on Spread The Sign | None found (only fingerspelling-letter datasets) | ⭐⭐⭐ Excellent | High |
| Croatian Sign Language | Yes — Spread The Sign + Hamburg SL Compendium | None found | ⭐⭐⭐ Excellent | High |
| Romanian Sign Language | Yes — Spread The Sign + RoCoISLR (9,000+ videos, [arXiv:2511.12767](https://arxiv.org/abs/2511.12767)) | None found — paper confirms the gap | ⭐⭐⭐ Excellent | High |
| Bulgarian Sign Language | Yes — Spread The Sign + Bulgarian Deaf Union dictionaries | None found | ⭐⭐⭐ Excellent | High |
| Ukrainian Sign Language | Yes — Spread The Sign | **Exists (small)** — UkrSL, ~2 hrs ([ACL 2026.unlp-1.6](https://aclanthology.org/2026.unlp-1.6/)) | **Not a candidate** | High |
| Georgian Sign Language | Some — geodeaf.ge online dictionary (~500 signs) | None found | ⭐⭐ Good | Medium |
| Armenian Sign Language | Weak/No — no public online lexicon confirmed | None found | **Not a candidate** (no usable isolated lexicon) | Medium |

### Latin America / Caribbean

| Language | Isolated Dictionary Exists? | Continuous Corpus Found? | Verdict | Confidence |
|---|---|---|---|---|
| Mexican Sign Language (LSM) | Yes — MX-ITESO-100, MSL-150, RGB-D corpus | Partial/None — only 30 short phrases; researchers note the gap | ⭐⭐⭐ Excellent | High |
| Colombian Sign Language (LSC) | Yes — LSC50, LSC70, LSC-54 | Partial, but exists — CoL-SLTD, 1,020 videos/39 sentences | ⭐ Weak | Medium |
| Peruvian Sign Language (LSP) | Yes — PUCP-DGI dictionary, LSP10 | **Yes** — PeruSIL/AEC (real interpreter footage) | **Not a candidate** | High |
| Chilean Sign Language (LSCh) | Some — SIL/OLAC wordlists | None found | ⭐⭐ Good | Medium |
| Argentine Sign Language (LSA) | Yes — LSA64 | **Yes** — LSA-T, 14,880 sentence videos ([arXiv:2211.15481](https://arxiv.org/abs/2211.15481)) | **Not a candidate** | High |
| Ecuadorian Sign Language | Some — LeSigLa_EC (275 words) | None found | ⭐⭐ Good | Medium |
| Venezuelan Sign Language | Some — print/online dictionary (Federación Venezolana de Sordos) | None found | ⭐⭐ Good (speculative) | Low |
| Caribbean Sign Language(s) | Composite category (Jamaican Country Sign, TTSL, etc.) — too vague to assess as one language | None found | Not assessable as a single category | Low |

## Data

See [`data.csv`](data.csv) for a machine-readable version of the master table (language, region, dictionary status, corpus status, verdict, confidence, key sources).

## License

This document compiles publicly available search findings for research-planning purposes. No sign language data itself is redistributed here — only citations to where such data may or may not exist. Verify all claims independently before relying on them.
