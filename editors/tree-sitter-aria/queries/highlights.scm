; Tree-sitter highlight queries for Aria.
; Capture names follow the standard nvim-treesitter convention.

; --- Comments ---------------------------------------------------------------
(line_comment) @comment

; --- Annotations ------------------------------------------------------------
(annotation
  "@" @attribute
  kind: (identifier) @attribute)

; --- Strings ----------------------------------------------------------------
(string) @string

; --- Bra / ket --------------------------------------------------------------
(ket) @string.special
(bra) @string.special

; --- Keywords ---------------------------------------------------------------
[
  "circuit"
  "observable"
] @keyword.function

[
  "qreg"
  "creg"
  "symbolic"
] @keyword.storage

[
  "let"
  "var"
] @keyword

[
  "repeat"
  "from"
  "to"
  "step"
  "when"
] @keyword.repeat

[
  "apply"
  "on"
  "oracle"
  "measure"
] @keyword.operator

; --- Constants --------------------------------------------------------------
(pi)      @constant.builtin
(boolean) @constant.builtin

; --- Numbers ----------------------------------------------------------------
(integer) @number
(float)   @number.float

; --- Gates (the gate_name node enumerates all of them) ----------------------
(gate_name) @function.builtin

; --- Type identifiers -------------------------------------------------------
(circuit_decl     name: (type_identifier) @type)
(observable_decl  name: (type_identifier) @type)

(param
  name: (identifier) @variable.parameter
  type: (type_identifier) @type)

; --- Function calls (sin, cos, bit, …) --------------------------------------
(call_expr callee: (identifier) @function.call)

; --- Variables --------------------------------------------------------------
(let_decl name: (identifier) @variable)
(var_decl name: (identifier) @variable)

(qreg_decl     name: (identifier) @variable.builtin)
(creg_decl     name: (identifier) @variable.builtin)
(symbolic_decl name: (identifier) @variable.builtin)

(indexed_register name: (identifier) @variable)

; --- Operators / punctuation ------------------------------------------------
[
  "->" "==" "!=" "<=" ">=" "<" ">"
  "+" "-" "*" "/" "^" "%"
  "="
] @operator

[
  "{" "}" "[" "]" "(" ")"
] @punctuation.bracket

[ "," ";" ":" ] @punctuation.delimiter
