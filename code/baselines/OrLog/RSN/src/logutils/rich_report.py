from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.pretty import Pretty

class Reporter:
    def __init__(self, log_dir: str = "LOGS_NEW", mode: str = "md") -> None:
        """
        Reporter for logging events. Supports 'html' or 'md' output modes.
        In 'html' mode, generates an HTML log with colored tables.
        In 'md' mode, generates a Markdown log with plain text and 0/1 gold flags.
        """
        self.retriever = None 
        Path(log_dir).mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mode = mode.lower()
        if self.mode == "html":
            self._out_path = Path(log_dir) / f"run_{stamp}.html"
            # record console with markup for HTML
            self.record_console = Console(record=True, width=120, markup=True)
        else:
            self._out_path = Path(log_dir) / f"run_{stamp}.md"
            # record console without markup for Markdown
            self.record_console = Console(record=True, width=120, markup=False)

        # Handler for recording console
        record_handler = RichHandler(
            console=self.record_console,
            markup=(self.mode == "html"),
            show_time=False,
            rich_tracebacks=True,
        )

        # Attach handler to root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[record_handler],
            format="%(message)s",
        )
        self.log = logging.getLogger("runner").info

    def save(self) -> None:
        """Save the recorded output to the appropriate file."""
        if self.mode == "html":
            self.record_console.save_html(self._out_path, inline_styles=True)
            msg = f"[green]Saved HTML log → {self._out_path}"
            self.record_console.print(msg)
            self.log("save_html")
        else:
            # Export plain text for Markdown
            text = self.record_console.export_text()
            with open(self._out_path, "w") as f:
                f.write(text)
            self.record_console.print(f"Saved MD log → {self._out_path}")
            self.log("save_md")

    def report(self, event: str, **payload: Any) -> None:
        """
        Dispatch to the correct handler based on event.
        """
        fn = getattr(self, f"_on_{event}", self._on_default)
        fn(payload)

    def _on_default(self, data: Dict) -> None:
        panel = Panel(Pretty(data, indent_guides=True), title="[yellow]EVENT[/]")
        self.record_console.print(panel)
        self.log(f"EVENT {list(data.keys())}")

    def _on_start_run(self, data: Dict) -> None:
        self.retriever = data.get("retriever").lower()

        title = f"[bold cyan]Starting run • Retr={self.retriever} • LLM={data['llm']} • method={data['method']} • ctx={data['context']}"
        self.record_console.rule(title)

    def _on_parse_query(self, data: Dict) -> None:
        parsed = data["parsed"]
        orig   = data.get("original_query", data.get("query", ""))

        tbl = Table(title="Parsed Query", show_header=True, header_style="magenta")
        tbl.add_column("Field")
        tbl.add_column("Value", overflow="fold")
        tbl.add_row("original query", orig)
        for key in ("atoms", "logical query"):
            if key in parsed:
                tbl.add_row(key, str(parsed[key]))
        self.record_console.print(tbl)

    def _on_entity_probabilities(self, data: Dict) -> None:
        tbl = Table(title=f"Probabilities for {data['entity']}", header_style="magenta")
        tbl.add_column("Atom")
        tbl.add_column("P(True)", justify="right")
        for atom, p in data["probabilities"].items():
            tbl.add_row(atom, f"{p:.4f}")
        self.record_console.print(tbl)

    def _on_prolog_program(self, data: Dict) -> None:
        prog_obj = data.get("program", "")
        prog_src = prog_obj if isinstance(prog_obj, str) else str(prog_obj)
        syntax = Syntax(prog_src, "prolog", theme="ansi_dark", line_numbers=True, word_wrap=True)
        panel  = Panel(syntax, title=f"Problog: {data.get('entity', '')}")
        self.record_console.print(panel)

    def _on_rankings(self, data: Dict) -> None:
        gold = set(data.get("gold", []))
        # Support multiple ranking types
        for which, title in [
            ("prob_rank", "ProbLog ranking"),
            ("bm25_rank", "BM25 ranking"),
            # (f"{self.retriever}_rank", f"{self.retriever.upper()} ranking"),
            ("e5_rank", "E5 ranking"),

            ("rag_rank",  "RAG ranking"),
        ]:
            if which not in data:
                continue

            ranking = data[which]

            tbl = Table(title=title)
            tbl.add_column("#", justify="right", width=3)
            tbl.add_column("Doc")
            tbl.add_column("Score", justify="right")
            tbl.add_column("Is_Gold", justify="center", width=7)
            for i, (doc, score) in enumerate(ranking, 1):
                is_gold = "1" if doc in gold else "0"
                if self.mode == "html":
                    style = "bold green" if doc in gold else None
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = float('nan')
                    tbl.add_row(str(i), doc, f"{score:.4f}", is_gold, style=style)
                    
                else:
                    # plain text, no color/style
                    # tbl.add_row(str(i), doc, f"{score:.4f}", is_gold)
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        print(f"[WARNING] The value was not a float! {score}")
                        score = float('nan')
                    tbl.add_row(str(i), doc, f"{score:.4f}", is_gold)
            self.record_console.print(tbl)

    def _on_query_index(self, data: Dict) -> None:
        self.record_console.print("\n" + "="*80 + "\n")
        msg = data.get("msg", "Query")
        tpl = data.get("template")
        if tpl:
            content = f"{msg}\n[italic]Template:[/] {tpl}"
        else:
            content = msg
        panel = Panel(content, title="Query Index", expand=False)
        self.record_console.print(panel)

    def _on_entity_context(self, data: Dict) -> None:
        syntax = Syntax(data["context"], "text", theme="ansi_dark", line_numbers=False, word_wrap=True)
        panel  = Panel(syntax, title=f"[cyan]Context for {data['entity']}[/cyan]")
        self.record_console.print(panel)

    def _on_final_table(self, data: Dict) -> None:
        df = data["df"]  # pandas.DataFrame
        tbl = Table(title="Consolidated Results", show_header=True, header_style="magenta")
        for idx_name in df.index.names:
            tbl.add_column(idx_name, overflow="fold")
        for col in df.columns:
            tbl.add_column(str(col), justify="right")
        for idx, row in df.iterrows():
            if isinstance(idx, tuple):
                idx_cells = [str(x) for x in idx]
            else:
                idx_cells = [str(idx)]
            row_cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
            tbl.add_row(*idx_cells, *row_cells)
        self.record_console.print(tbl)
        self.log("final_table")

    def _on_rag_output(self, data: Dict) -> None:
        syntax = Syntax(data.get("output", ""), "text", theme="ansi_dark",
                        line_numbers=False, word_wrap=True)
        panel  = Panel(syntax, title=f"[cyan]RAG LLM output for {data.get('entity', '')}[/cyan]")
        self.record_console.print(panel)
