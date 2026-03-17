import csv
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

PROMPT_TEMPLATE = r"""
<ROLE>
You are a senior systematic review screening assistant with expertise in evidence synthesis. You apply eligibility criteria strictly and consistently. You have over 3 years of experience in incident management research, AI, and LLM-based systems. 
Audience: researchers performing title and abstract screening. 
Perspective: methodological rigor and transparency.
</ROLE>

<TASK>
Screen one article using only the provided title and abstract. Determine overall eligibility (YES or NO). This is Phase 1 (P1) screening (title + abstract): NO means the article will not proceed to Phase 2 (full-text reading), YES means it will.
Write an Explanation as a single paragraph (~100 words), grounded strictly in the title/abstract. Do NOT invent information. First check exclusion criteria, then inclusion criteria.
If ANY exclusion criterion applies, return NO. In the Explanation, start with "NO - ECX: " (where X is the number of the single most relevant exclusion criterion), then continue with the justification.
If IC1 is not met, return NO. In the explanation: "NO - IC1: ". For IC2: if met, continue. If not met, return NO (in the explanation: "NO - IC2: ").
If all inclusion criteria are met and no exclusion criteria apply, return YES. Start in the explanation with "YES: " followed by the remaining justification.
</TASK>

<CONTEXT>
Detection involves identifying anomalies or incidents within a system, serving as the first step in the incident management process. Triage is a redirect of an incident to a specific sector or group of people to determine who will resolve which incident. Root Cause Analysis (RCA) focuses on diagnosing the underlying causes of incidents, allowing teams to address problems effectively and prevent recurrence. Remediation refers to the process of resolving or mitigating incidents, ensuring that systems return to normal operations after an issue has been identified and analysed. Post-incident or Postmortem refers to the period after the incident is resolved, involving the post-incident report and reflection and analysis of what could be improved and done differently. For inclusion criteria: IC1 requires that the study addresses any kind of incident management using artificial intelligence, even if the term AIOps is not explicitly used. The scope of incident management here is software (software incident management), so exclude automotive, chemical, biological, hardware, etc. contexts. Incident management touches on other areas but only comes into play if the focus is on software incidents in general. IC2 further requires that the study proposes, evaluates, or discusses remediation, mitigation, or solution strategies for incidents using large language models. Studies focused exclusively on detection do not meet IC2, while root cause analysis studies may be included only if they indicate remediation strategies based on LLMs that seek to resolve the identified cause. Studies focused on RCA that do not even mention remediation steps or actions do not attend IC2 either. RCA can include remediation because some authors describe suggest mitigation actions, so read carefully so as not to incorrectly exclude it. Remediation appears with synonyms, so map equivalent terms such as remediation, mitigation, resolution, solution, diagnosis, fault resolution, etc. If the content does not indicate a way to solve or remediate an incident, it should not be included. Incident solutions or mitigation of these incidents can still occur even if the AIOps stages are not clearly defined or appear in a different contexts, as long as there is remediation using LLMs.
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
5) EC 5 (The full text is not accessible)
6) EC 6 (The study is a duplicate of another already identified work)

Article:
Title: {title}
Abstract: {abstract}
</INPUT_FORMAT>

<OUTPUT_FORMAT>
YES|NO

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


PRICING_PER_1M = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}


@dataclass
class CallMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class RunStats:
    model: str = ""
    start_time: str = ""
    end_time: str = ""
    total_calls: int = 0
    yes_count: int = 0
    no_count: int = 0
    calls: list[CallMetrics] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.total_calls if self.total_calls else 0.0

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / self.total_calls if self.total_calls else 0.0

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.total_calls if self.total_calls else 0.0

    @property
    def total_reasoning_tokens(self) -> int:
        return sum(c.reasoning_tokens for c in self.calls)

    @property
    def avg_throughput_tokens_per_s(self) -> float:
        if not self.calls:
            return 0.0
        rates = [c.output_tokens / c.latency_s for c in self.calls if c.latency_s > 0]
        return statistics.mean(rates) if rates else 0.0

    @property
    def median_latency_s(self) -> float:
        return statistics.median(c.latency_s for c in self.calls) if self.calls else 0.0

    @property
    def p95_latency_s(self) -> float:
        if not self.calls:
            return 0.0
        sorted_lat = sorted(c.latency_s for c in self.calls)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


def call_llm(prompt: str, cfg: LLMConfig) -> tuple[str, CallMetrics]:
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url or None,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )

    start = time.perf_counter()
    response = client.responses.create(
        model=cfg.model, input=prompt, text={"verbosity": "low"}
    )
    elapsed = time.perf_counter() - start

    usage = response.usage
    output_details = getattr(usage, "output_tokens_details", None) if usage else None
    metrics = CallMetrics(
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
        latency_s=elapsed,
    )

    return response.output_text or "", metrics


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> tuple[float, float, float]:
    prices = PRICING_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    input_cost = input_tokens / 1_000_000 * prices["input"]
    output_cost = output_tokens / 1_000_000 * prices["output"]
    return input_cost, output_cost, input_cost + output_cost


def write_stats_report(path: str, stats: RunStats) -> None:
    input_cost, output_cost, total_cost = estimate_cost(
        stats.model, stats.total_input_tokens, stats.total_output_tokens
    )

    lines = [
        f"Modelo: {stats.model}",
        f"Início: {stats.start_time}",
        f"Fim: {stats.end_time}",
        "",
        f"Total de chamadas: {stats.total_calls}",
        f"Artigos aceitos (YES): {stats.yes_count}",
        f"Artigos rejeitados (NO): {stats.no_count}",
        "",
        "--- Tokens ---",
        f"Input tokens  (total):             {stats.total_input_tokens:,}",
        f"Output tokens (total):             {stats.total_output_tokens:,}",
        f"  Reasoning tokens (total):        {stats.total_reasoning_tokens:,}",
        f"Tokens totais:                     {stats.total_tokens:,}",
        f"Input tokens  (média por chamada): {stats.avg_input_tokens:,.1f}",
        f"Output tokens (média por chamada): {stats.avg_output_tokens:,.1f}",
        "",
        "--- Custo estimado (USD) ---",
        f"Input:  ${input_cost:,.4f}",
        f"Output: ${output_cost:,.4f}",
        f"Total:  ${total_cost:,.4f}",
        "",
        "--- Latência ---",
        f"Tempo total:             {stats.total_latency_s:,.2f}s",
        f"Tempo médio por chamada: {stats.avg_latency_s:,.2f}s",
        f"Mediana:                 {stats.median_latency_s:,.2f}s",
        f"Percentil 95 (p95):      {stats.p95_latency_s:,.2f}s",
        f"Menor latência:          {min((c.latency_s for c in stats.calls), default=0):,.2f}s",
        f"Maior latência:          {max((c.latency_s for c in stats.calls), default=0):,.2f}s",
        "",
        "--- Throughput ---",
        f"Output tokens/s (média): {stats.avg_throughput_tokens_per_s:,.1f}",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_verdict(content: str) -> tuple[str, str]:
    raw = (content or "").strip()
    m = re.search(r"^(YES|NO)\b", raw, flags=re.IGNORECASE | re.MULTILINE)
    verdict = m.group(1).upper() if m else "NO"
    m2 = re.search(r"^Explanation:\s*(.*)$", raw, flags=re.IGNORECASE | re.MULTILINE)
    explanation = m2.group(1).strip() if m2 else ""
    explanation_line = f"Explanation: {explanation}" if explanation else ""
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

    input_csv = "./selection_p1/avaliar_ia.csv"

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

    col_id = fieldnames[0] if fieldnames else "ID"
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

    stats = RunStats(
        model=cfg.model,
        start_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    for idx, row in enumerate(rows, 1):
        already_done = any(
            (row.get(c, "") or "").strip() for c in (col_extract, col_reject, col_just)
        )
        if already_done:
            continue

        title = (row.get(col_title, "") or "").strip()
        abstract = (row.get(col_abstract, "") or "").strip()
        prompt = PROMPT_TEMPLATE.format(title=title, abstract=abstract)

        content, metrics = call_llm(prompt, cfg)
        stats.calls.append(metrics)
        stats.total_calls += 1

        verdict, explanation_line = parse_verdict(content)

        if verdict == "YES":
            row[col_extract] = "Aceito"
            row[col_reject] = ""
            stats.yes_count += 1
        else:
            row[col_extract] = "Rejeitado"
            codes = parse_reason_codes(explanation_line)
            primary = next(
                (ec_map[c] for c in codes if c.startswith("EC") and c in ec_map), ""
            ) or next(
                (ic_map[c] for c in codes if c.startswith("IC") and c in ic_map), ""
            )
            row[col_reject] = primary
            stats.no_count += 1

        row[col_just] = explanation_line

        write_csv(input_csv, fieldnames, rows)
        row_id = (row.get(col_id, "") or "").strip()
        prefix = f"[{idx}/{len(rows)}]{' ID=' + row_id if row_id else ''}"
        print(f"{prefix} -> {verdict} : {title}", flush=True)

    stats.end_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if stats.total_calls > 0:
        report_path = "./selection_p1/stats_report.txt"
        write_stats_report(report_path, stats)
        print(f"\nRelatório de estatísticas salvo em: {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
