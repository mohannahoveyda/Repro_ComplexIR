
import re
from typing import List, Dict


# ---------- Boolean Expression AST ----------
class Expr:
    def to_prolog(self, mapping, ent):
        raise NotImplementedError

class AtomExpr(Expr):
    def __init__(self, name): self.name = name
    def to_prolog(self, mapping, ent):
        return f"{mapping[self.name]}({ent})"

class NotExpr(Expr):
    def __init__(self, operand): self.operand = operand
    def to_prolog(self, mapping, ent):
        return f"\\+({self.operand.to_prolog(mapping, ent)})"

class AndExpr(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def to_prolog(self, mapping, ent):
        return f"({self.left.to_prolog(mapping, ent)},{self.right.to_prolog(mapping, ent)})"

class OrExpr(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def to_prolog(self, mapping, ent):
        return f"({self.left.to_prolog(mapping, ent)};{self.right.to_prolog(mapping, ent)})"

# ---------- Tokenizer & Recursive‑descent Parser ----------
class BoolParser:
    def __init__(self, atoms):
        # sort atoms by length so longest matches first
        self.atoms = sorted(atoms, key=len, reverse=True)
        atom_pat = "|".join(re.escape(a) for a in self.atoms)
        self.token_re = re.compile(rf"\s*({atom_pat}|AND|OR|NOT|\(|\))\s*", re.I)

    def tokenize(self, expr):
        toks = self.token_re.findall(expr)
        if not toks:
            raise ValueError(f"Cannot tokenize: {expr!r}")
        return [t.strip() for t in toks]

    def parse(self, expr: str) -> Expr:
        # --- 1) sanitize: strip trailing periods/dots and whitespace ---
        expr = expr.strip()
        if expr.endswith("."):
            expr = expr[:-1].rstrip()

        # --- 2) balance parentheses if they’re off by a bit ---
        opens  = expr.count("(")
        closes = expr.count(")")
        if opens > closes:
            expr = expr + ")" * (opens - closes)
        elif closes > opens:
            expr = "(" * (closes - opens) + expr

        # optional: log the sanitized expression
        # reporter.report("debug", msg=f"Sanitized Boolean expr: {expr}")


        self.tokens = self.tokenize(expr)
        self.pos    = 0
        node       = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f"Extra token after parsing: {self.tokens[self.pos]!r}")
        return node

    def _parse_or(self):
        left = self._parse_and()
        while self._peek_upper() == "OR":
            self._next()
            left = OrExpr(left, self._parse_and())
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._peek_upper() == "AND":
            self._next()
            left = AndExpr(left, self._parse_not())
        return left

    def _parse_not(self):
        if self._peek_upper() == "NOT":
            self._next()
            return NotExpr(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self):
        tok = self._peek()
        if tok == "(":
            self._next()
            node = self._parse_or()
            if self._peek() != ")":
                raise ValueError("Missing ')'")
            self._next()
            return node
        if tok in self.atoms:
            self._next()
            return AtomExpr(tok)
        raise ValueError(f"Unknown atom: {tok!r}")

    def _peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def _peek_upper(self): return (self._peek() or "").upper()
    def _next(self): 
        t = self._peek(); self.pos += 1; return t
