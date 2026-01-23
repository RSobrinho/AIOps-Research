import csv
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

PROMPT_TEMPLATE = r"""
<ROLE>
You are a senior systematic review screening assistant with expertise in evidence synthesis. You apply eligibility criteria strictly and consistently. You have over 10 years of experience in incident management research, AI, and LLM-based systems. Audience: researchers performing title and abstract screening. Perspective: methodological rigor and transparency.
</ROLE>

<TASK>
Screen one article using only the provided title and abstract. Determine overall eligibility (YES, NO, or MAYBE). Then, write an Explanation as a single paragraph (~100 words), grounded strictly in the title/abstract. First check exclusion criteria. 
If ANY exclusion criterion applies, return NO and stop; in the Explanation, start with "NO - ECX: " (where X is just an example, put the number of the correspondent EC or IC) using ONLY ONE (the single most relevant exclusion criterion), then continue with the justification. If no exclusion criterion applies, check inclusion criteria.
If IC1 is not met, return NO; in the Explanation, start with "NO - IC1: " then continue with the justification. For IC2 (remediation/mitigation/solution using LLMs): if it is clearly met, continue; if it is clearly NOT met, return NO (start "NO - IC2: "); if it is UNCERTAIN but there are indications it might be met (based on title/abstract), return MAYBE (start "MAYBE - IC2") because full text would be needed to confirm remediation. 
If all inclusion criteria are met and no exclusion criteria apply, return YES; the Explanation must start with "YES: " and then continue with the justification.
Use the information existent inside title and abstract. Do NOT invent. Think deeply about this.
</TASK>

<CONTEXT>
Screening should start with the exclusion criteria to quickly remove studies that should not proceed, prioritizing secondary studies, non peer reviewed works, theses, non English papers, inaccessible full texts, or duplicates. After applying the exclusion criteria, confirm the basic requirements, that the study is primary, peer reviewed, and written in English. Only then apply the inclusion criteria, noting that all inclusion criteria must be satisfied. IC1 requires that the study addresses incident management using artificial intelligence, even if the term AIOps is not explicitly used. IC2 further requires that the study proposes, evaluates, or discusses remediation, mitigation, or solution strategies for incidents using large language models, explicitly targeting the remediation stage of AIOps. In this context, AIOps is understood as comprising three core stages, detection, root cause analysis, and remediation. Therefore, the solution must aim to address and resolve the root cause identified during the RCA stage, acting as a concrete or proposed intervention to solve the problem initially detected during the detection phase. Studies focused exclusively on detection do not meet IC2, while root cause analysis studies may be included only if they indicate solutions or remediation strategies based on LLMs that seek to resolve the identified cause.
</CONTEXT>

<INPUT_FORMAT>
Inclusion criteria:
1) IC 1 (The study addresses the topic of incident management using artificial intelligence.)
2) IC 2 (The study presents or evaluates a remediation, mitigation, or solution proposal for an incident using large language models (LLMs).)

Exclusion criteria:
1) EC 1 (The study is a secondary study such as a review, mapping study, or survey)
2) EC 2 (The study was not peer reviewed (grey literature))
3) EC 3 (The work is a thesis or dissertation)
4) EC 4 (The study is not written in English)
5) EC 5  (The full text is not accessible)
6) EC 6 (The study is a duplicate of another already identified work)


Article:
Title: {title}
Abstract: {abstract}
</INPUT_FORMAT>

<OUTPUT_FORMAT>
YES|NO|MAYBE

Explanation: [PUT HERE THE CRITERIA AND THE EXPLANATION FORMATED]
</OUTPUT_FORMAT>

<QUALITY_STRATEGY>
Before responding, internally define what a 10/10 screening output looks like for this task. Then internally verify consistency across overall, inclusion, exclusion, and explanation. Show only the final output, no internal notes.
</QUALITY_STRATEGY>
""".strip()


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: int = 60
    max_retries: int = 5
    retry_backoff_s: float = 2.0


def call_llm(prompt: str, cfg: LLMConfig) -> str:
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url or None,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )

    response = client.responses.create(
        model=cfg.model, input=prompt, text={"verbosity": "low"}
    )

    return response.output_text or ""


def parse_verdict(content: str) -> tuple[str, str]:
    raw = (content or "").strip()
    m = re.search(r"^(YES|NO|MAYBE)\b", raw, flags=re.IGNORECASE | re.MULTILINE)
    verdict = m.group(1).upper() if m else "NO"
    m2 = re.search(r"^Explanation:\s*(.*)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    explanation = m2.group(1).strip() if m2 else ""
    explanation_line = f"Explanation: {explanation}".strip()
    return verdict, explanation_line


def parse_reason_codes(explanation_line: str) -> list[str]:
    codes = re.findall(r"\b(?:IC|EC)\s*\d+\b", explanation_line, flags=re.IGNORECASE)
    out: list[str] = []
    for c in codes:
        c = c.upper().replace(" ", "")
        if c not in out:
            out.append(c)
    return out


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    load_dotenv()

    input_csv = "./selection_p1/artigos.csv"

    cfg = LLMConfig(
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        api_key=os.getenv("LLM_API_KEY"),
        timeout_s=60,
        max_retries=5,
        retry_backoff_s=2.0,
    )

    with open(input_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    col_title = "Título do Artigo"
    col_abstract = "Abstract"
    col_extract = "Vai para a etapa de extração?"
    col_reject = "Rejeitado/Não incluído, por quê?"
    col_just = "Justificativa"

    ic_map = {
        "IC1": "IC 1 (O trabalho aborda o tópico de gerenciamento de incidentes utilizando inteligência artificial.)",
        "IC2": "IC 2 (O trabalho apresenta ou avalia uma proposta de remediação, mitigação ou solução de um incidente utilizando LLM.)",
    }
    ec_map = {
        "EC1": "EC 1 (É um estudo secundário como revisão, mapeamento ou survey)",
        "EC2": "EC 2 (O trabalho não foi revisado por pares (literatura cinza))",
        "EC3": "EC 3 (É uma tese ou dissertação)",
        "EC4": "EC 4 (Não está em inglês)",
        "EC5": "EC 5  (Texto completo não acessível)",
        "EC6": "EC 6 (Duplicata de outro trabalho já existente)",
    }

    if col_extract not in fieldnames:
        fieldnames.append(col_extract)
    if col_reject not in fieldnames:
        fieldnames.append(col_reject)
    if col_just not in fieldnames:
        fieldnames.append(col_just)

    for r in rows:
        r.setdefault(col_extract, "")
        r.setdefault(col_reject, "")
        r.setdefault(col_just, "")

    for idx, row in enumerate(rows, 1):
        if (
            row.get(col_extract, "").strip()
            or row.get(col_reject, "").strip()
            or row.get(col_just, "").strip()
        ):
            continue

        title = (row.get(col_title, "") or "").strip()
        abstract = (row.get(col_abstract, "") or "").strip()
        prompt = PROMPT_TEMPLATE.format(title=title, abstract=abstract)

        content = call_llm(prompt, cfg)
        verdict, explanation_line = parse_verdict(content)
        codes = (
            parse_reason_codes(explanation_line) if verdict in ("NO", "MAYBE") else []
        )

        row[col_extract] = (
            "Aceito"
            if verdict == "YES"
            else ("Em dúvida" if verdict == "MAYBE" else "Rejeitado")
        )
        if verdict in ("NO", "MAYBE"):
            primary = ""
            for c in codes:
                if c.startswith("EC") and c in ec_map:
                    primary = ec_map[c]
                    break
            if not primary:
                for c in codes:
                    if c.startswith("IC") and c in ic_map:
                        primary = ic_map[c]
                        break
            row[col_reject] = primary
        else:
            row[col_reject] = ""
        row[col_just] = explanation_line

        write_csv(input_csv, fieldnames, rows)
        print(f"[{idx}/{len(rows)}] -> {verdict} : {title}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
