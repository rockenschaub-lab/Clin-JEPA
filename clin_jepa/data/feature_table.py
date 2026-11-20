"""Write the shared WeavEHR feature inventory from ``configs/data/features.yaml``."""

import argparse

from clin_jepa.utils import ensure_dir, expand_path, load_config


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def allowbreaks(s: str) -> str:
    return s.replace(".", r".\allowbreak ").replace(r"\_", r"\_\allowbreak ")


def variables_cell(domain: str, info: dict) -> str:
    return ", ".join(latex_escape(v) for v in info["value_cols"])


def xltabular(caption: str, label: str, sources: dict, font_size: str) -> list[str]:
    header = r"\textbf{Concept} & \textbf{Unit} & \textbf{Category} & \textbf{Name in text} \\"
    lines = [
        "{" + font_size,
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{xltabular}{\linewidth}{@{}"
        r">{\raggedright\arraybackslash}p{0.16\linewidth} "
        r">{\raggedright\arraybackslash}p{0.25\linewidth} "
        r">{\raggedright\arraybackslash}p{0.14\linewidth} "
        r">{\raggedright\arraybackslash}X@{}}",
        rf"\caption{{{caption}}} \label{{{label}}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\multicolumn{4}{c}{\textit{(\tablename~\thetable\ continued)}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{4}{r}{\textit{Continued on next page\ldots}} \\",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
        "",
    ]
    for domain, info in sources.items():
        cells = [latex_escape(domain), latex_escape(info.get("source", "—")), latex_escape(info.get("category", "—"))]
        lines.append(
            f"{allowbreaks(cells[0])} & \\texttt{{{allowbreaks(cells[1])}}} & "
            f"{allowbreaks(cells[2])} & {allowbreaks(variables_cell(domain, info))} \\\\"
        )
    return lines + [r"\end{xltabular}", "}", ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_config", default="configs/data/features.yaml")
    parser.add_argument("--output_dir", default="${CLIN_JEPA_OUTPUT}/tables")
    args = parser.parse_args()

    features = load_config(args.features_config)

    def inventory(rules):
        return {
            r["concept"]: {"source": r.get("output_unit", r["unit"]) or "—", "category": r["category"],
                           "value_cols": [r["display_name"]]}
            for r in rules
        }

    lines = xltabular(
        "Shared WeavEHR observation concepts. Actual mapping coverage is reported per dataset.",
        "tab:obs_features",
        inventory(features["observations"]),
        r"\footnotesize",
    ) + xltabular(
        "Shared WeavEHR intervention concepts. Unavailable concepts are not imputed.",
        "tab:action_features",
        inventory(features["actions"]),
        r"\scriptsize",
    )
    path = ensure_dir(expand_path(args.output_dir)) / "appendix_features_table.tex"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
