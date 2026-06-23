" Vim/Neovim syntax for the Aria quantum DSL.
" Reference grammar: crates/aria-core/src/ast/aria.rs (parse_aria).

if exists("b:current_syntax")
  finish
endif

syntax case match

" --- Comments ----------------------------------------------------------------
syntax match ariaQuantumComment "--.*$" contains=@Spell

" --- Annotations -------------------------------------------------------------
syntax match ariaQuantumAnnotation "@\(assert\|prove\|bound\|resource_bound\|ensure\|requires\|invariant\|note\)\>"
syntax match ariaQuantumAnnotationOther "@[a-zA-Z_][a-zA-Z0-9_]*"

" --- Strings -----------------------------------------------------------------
syntax region ariaQuantumString start=+"+ skip=+\\"+ end=+"+ contains=ariaQuantumEscape
syntax match  ariaQuantumEscape "\\." contained

" --- Bra / ket ---------------------------------------------------------------
syntax match ariaQuantumKet "|[+\-0-9a-zA-Z_]*>"
syntax match ariaQuantumBra "<[+\-0-9a-zA-Z_]*|"

" --- Keywords ----------------------------------------------------------------
syntax keyword ariaQuantumDeclaration circuit observable
syntax keyword ariaQuantumRegister    qreg creg symbolic
syntax keyword ariaQuantumBinding     let var
syntax keyword ariaQuantumControl     repeat from to step when
syntax keyword ariaQuantumApplication apply on oracle measure

syntax keyword ariaQuantumConstant pi true false

" --- Builtin functions (only at call sites) ----------------------------------
syntax match ariaQuantumBuiltin "\<\(sin\|cos\|tan\|sqrt\|exp\|log\|abs\|floor\|ceil\|bit\|min\|max\)\>\ze("

" --- Gates -------------------------------------------------------------------
syntax keyword ariaQuantumGateMulti    CCX TOFFOLI CSWAP FREDKIN CNOT SWAP CX CY CZ CP RXX RYY RZZ
syntax keyword ariaQuantumGateRotation RX RY RZ U1 U3 U P
syntax keyword ariaQuantumGateSingle   SDG TDG SX H X Y Z S T I ID

" --- Numbers -----------------------------------------------------------------
syntax match ariaQuantumFloat   "\<[0-9]\+\.[0-9]\+\([eE][+-]\?[0-9]\+\)\?\>"
syntax match ariaQuantumInteger "\<[0-9]\+\>"

" --- Type annotation `name : type` ------------------------------------------
syntax match ariaQuantumType "\(:\s*\)\@<=[a-zA-Z_][a-zA-Z0-9_]*"

" --- Circuit/observable name in declaration ---------------------------------
syntax match ariaQuantumCircuitName "\(\<\(circuit\|observable\)\s\+\)\@<=[A-Z][a-zA-Z0-9_]*"

" --- Operators ---------------------------------------------------------------
syntax match ariaQuantumArrow      "->"
syntax match ariaQuantumOpCompare  "==\|!=\|<=\|>=\|<\|>"
syntax match ariaQuantumOpAssign   "="
syntax match ariaQuantumOpArith    "[+\-*/^%]"

" --- Highlight links ---------------------------------------------------------
highlight default link ariaQuantumComment           Comment
highlight default link ariaQuantumAnnotation        PreProc
highlight default link ariaQuantumAnnotationOther   PreProc
highlight default link ariaQuantumString            String
highlight default link ariaQuantumEscape            SpecialChar
highlight default link ariaQuantumKet               Special
highlight default link ariaQuantumBra               Special
highlight default link ariaQuantumDeclaration       Structure
highlight default link ariaQuantumRegister          StorageClass
highlight default link ariaQuantumBinding           Keyword
highlight default link ariaQuantumControl           Repeat
highlight default link ariaQuantumApplication       Statement
highlight default link ariaQuantumConstant          Constant
highlight default link ariaQuantumBuiltin           Function
highlight default link ariaQuantumGateMulti         Type
highlight default link ariaQuantumGateRotation      Type
highlight default link ariaQuantumGateSingle        Type
highlight default link ariaQuantumFloat             Float
highlight default link ariaQuantumInteger           Number
highlight default link ariaQuantumType              Type
highlight default link ariaQuantumCircuitName       Title
highlight default link ariaQuantumArrow             Operator
highlight default link ariaQuantumOpCompare         Operator
highlight default link ariaQuantumOpAssign          Operator
highlight default link ariaQuantumOpArith           Operator

let b:current_syntax = "aria_quantum"
