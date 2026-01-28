import argparse
import contextlib
import io
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


RESULTS_DIR = Path("results/health/soft_max_trades_verify")
CASE_OUTPUTS = {
    "case1": RESULTS_DIR / "case1.out.txt",
    "case2": RESULTS_DIR / "case2.out.txt",
    "case3": RESULTS_DIR / "case3.out.txt",
}


@dataclass
class CaseResult:
    name: str
    ok: bool
    details: List[str]
    stdout: str
    stderr: str
    output_path: Path


def _run_internal_case() -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from Live.risk_manager import RiskManager

        rm = RiskManager(
            max_trades=5,
            soft_trades_after=1,
            conservative_risk_mult=0.5,
            max_consecutive_losses=999,
            cooldown_on_loss_sec=0,
            cooldown_on_streak_sec=0,
        )

        required_attrs = [
            "conservative_mode",
            "conservative_reason",
            "risk_mult",
            "conservative_risk_mult",
        ]
        missing = [attr for attr in required_attrs if not hasattr(rm, attr)]
        if missing:
            available = sorted(vars(rm).keys())
            raise AttributeError(
                f"Missing RiskManager attributes: {missing}. Available: {available}"
            )

        rm.register_trade("EXIT")
        rm.register_trade("EXIT")

        print(
            "STATE "
            f"conservative_mode={rm.conservative_mode} "
            f"reason={rm.conservative_reason} "
            f"risk_mult={rm.risk_mult} "
            f"conservative_risk_mult={rm.conservative_risk_mult}"
        )

    sys.stdout.write(buf.getvalue())
    return 0


def _parse_state(stdout: str) -> Tuple[Dict[str, str], List[str]]:
    state_line = None
    for line in stdout.splitlines():
        if line.startswith("STATE "):
            state_line = line
    if state_line is None:
        return {}, ["Missing STATE line in stdout."]

    parts = state_line.replace("STATE ", "", 1).split()
    state = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        state[key] = value
    return state, []


def _validate_case(
    name: str,
    stdout: str,
    stderr: str,
    output_path: Path,
    expect_disabled_log: bool,
    expect_conservative: bool,
    expect_reason_soft: bool,
    expect_risk_match: bool,
) -> CaseResult:
    details: List[str] = []
    disabled_text = "SOFT_MAX_TRADES disabled (PIPELINE)"
    disabled_count = stdout.count(disabled_text)

    if expect_disabled_log and disabled_count != 1:
        details.append(f"Expected 1 disabled log, got {disabled_count}.")
    if not expect_disabled_log and disabled_count != 0:
        details.append(f"Expected 0 disabled logs, got {disabled_count}.")

    state, state_errors = _parse_state(stdout)
    if state_errors:
        details.extend(state_errors)
    else:
        required_keys = {
            "conservative_mode",
            "reason",
            "risk_mult",
            "conservative_risk_mult",
        }
        missing_keys = sorted(required_keys - set(state.keys()))
        if missing_keys:
            details.append(
                f"Missing state keys {missing_keys}. Found keys: {sorted(state.keys())}."
            )
        else:
            conservative_mode = state["conservative_mode"].lower() == "true"
            reason = state["reason"]
            risk_mult = state["risk_mult"]
            conservative_risk_mult = state["conservative_risk_mult"]

            if expect_conservative != conservative_mode:
                details.append(
                    f"Expected conservative_mode={expect_conservative}, got {conservative_mode}."
                )

            if expect_reason_soft and reason != "SOFT_MAX_TRADES":
                details.append(
                    f"Expected conservative_reason=SOFT_MAX_TRADES, got {reason or '<empty>'}."
                )
            if not expect_reason_soft and reason == "SOFT_MAX_TRADES":
                details.append("Expected conservative_reason != SOFT_MAX_TRADES.")

            risk_match = risk_mult == conservative_risk_mult
            if expect_risk_match != risk_match:
                details.append(
                    "Expected risk_mult "
                    f"{'==' if expect_risk_match else '!='} "
                    f"conservative_risk_mult, got {risk_mult} vs {conservative_risk_mult}."
                )

    if stderr.strip():
        details.append(f"stderr had output: {stderr.strip()}")

    return CaseResult(
        name=name,
        ok=not details,
        details=details,
        stdout=stdout,
        stderr=stderr,
        output_path=output_path,
    )


def _run_case(name: str, env_overrides: Dict[str, str]) -> Tuple[str, str, int]:
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [
        sys.executable,
        "-m",
        "analysis.verify_soft_max_trades_patch",
        "--internal-case",
        name,
    ]
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def _find_flag_occurrences() -> List[str]:
    matches: List[str] = []
    bat_dir = Path("bat")
    if not bat_dir.exists():
        return matches

    for root, _, files in os.walk(bat_dir):
        for filename in files:
            path = Path(root) / filename
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "PIPELINE_DISABLE_SOFT_MAX_TRADES" in content:
                matches.append(str(path))
    return sorted(matches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-case", dest="internal_case")
    args = parser.parse_args()

    if args.internal_case:
        try:
            return _run_internal_case()
        except Exception as exc:
            print(f"INTERNAL_CASE_ERROR: {exc}")
            return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "case1",
            {"RUN_MODE": "PIPELINE", "PIPELINE_DISABLE_SOFT_MAX_TRADES": "1"},
            True,
            False,
            False,
            False,
        ),
        (
            "case2",
            {"RUN_MODE": "PIPELINE"},
            False,
            True,
            True,
            True,
        ),
        (
            "case3",
            {"RUN_MODE": "LIVE", "PIPELINE_DISABLE_SOFT_MAX_TRADES": "1"},
            False,
            True,
            True,
            True,
        ),
    ]

    results: List[CaseResult] = []
    for name, env, expect_disabled, expect_conservative, expect_reason, expect_risk_match in cases:
        stdout, stderr, returncode = _run_case(name, env)
        output_path = CASE_OUTPUTS[name]
        output_path.write_text(stdout, encoding="utf-8")
        if returncode != 0:
            details = [
                f"Subprocess exit code {returncode}.",
                "See stderr for details." if stderr.strip() else "No stderr output.",
            ]
            results.append(
                CaseResult(
                    name=name,
                    ok=False,
                    details=details,
                    stdout=stdout,
                    stderr=stderr,
                    output_path=output_path,
                )
            )
            continue

        results.append(
            _validate_case(
                name=name,
                stdout=stdout,
                stderr=stderr,
                output_path=output_path,
                expect_disabled_log=expect_disabled,
                expect_conservative=expect_conservative,
                expect_reason_soft=expect_reason,
                expect_risk_match=expect_risk_match,
            )
        )

    occurrences = _find_flag_occurrences()

    report_lines: List[str] = []
    overall_ok = all(result.ok for result in results)
    report_lines.append(f"SOFT_MAX_TRADES verify: {'PASS' if overall_ok else 'FAIL'}")
    report_lines.append("")
    for result in results:
        report_lines.append(f"{result.name.upper()}: {'PASS' if result.ok else 'FAIL'}")
        report_lines.append(f"stdout: {result.output_path}")
        if result.details:
            for detail in result.details:
                report_lines.append(f"  - {detail}")
        report_lines.append("")

    report_lines.append("PIPELINE_DISABLE_SOFT_MAX_TRADES occurrences in bat/:")
    if occurrences:
        for path in occurrences:
            report_lines.append(f"  - {path}")
    else:
        report_lines.append("  (none found)")

    report_path = RESULTS_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
