/**
 * tree-sitter grammar for the Aria quantum DSL.
 *
 * Mirrors the production rules in
 *   crates/aria-core/src/ast/aria.rs (parse_aria).
 *
 * Tested against every file in examples/aria/ via the test corpus
 * under test/corpus/.
 */

const PREC = {
  unary:    8,  // -x
  power:    7,  // x^y       right-assoc
  product:  6,  // *  /  %
  sum:      5,  // +  -
  compare:  4,  // == != < <= > >=
  call:     9,
};

module.exports = grammar({
  name: 'aria',

  extras: $ => [/\s/, $.line_comment],
  word: $ => $.identifier,

  rules: {
    source_file: $ => repeat(choice(
      $.annotation,
      $.circuit_decl,
      $.observable_decl,
    )),

    // --- Comments -----------------------------------------------------------
    line_comment: _ => token(seq('--', /.*/)),

    // --- Annotations --------------------------------------------------------
    annotation: $ => seq(
      '@',
      field('kind', $.identifier),
      optional(field('content', $._annotation_content)),
    ),

    // Keep annotation contents loose: tokens up to next newline,
    // optionally a brace-delimited block.
    _annotation_content: $ => choice(
      $._annotation_inline,
      $.annotation_block,
    ),
    _annotation_inline: $ => repeat1(choice(
      $.identifier, $.integer, $.float, $.string, $.ket, $.bra,
      '=', '+', '-', '*', '/', '^', '%', ',', '(', ')',
    )),
    annotation_block: $ => seq(
      '{',
      repeat(choice(
        $.identifier, $.integer, $.float, $.string, $.ket, $.bra,
        '=', '+', '-', '*', '/', '^', '%', ',', '(', ')', '|',
      )),
      '}',
    ),

    // --- Top-level declarations --------------------------------------------
    circuit_decl: $ => seq(
      'circuit',
      field('name', $.type_identifier),
      optional(field('params', $.param_list)),
      field('body', $.block),
    ),

    observable_decl: $ => seq(
      'observable',
      field('name', $.type_identifier),
      field('body', $.block),
    ),

    param_list: $ => seq(
      '(',
      optional(seq($.param, repeat(seq(',', $.param)))),
      ')',
    ),

    param: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $.type_identifier),
    ),

    block: $ => seq('{', repeat($._statement), '}'),

    // --- Statements ---------------------------------------------------------
    _statement: $ => choice(
      $.qreg_decl,
      $.creg_decl,
      $.symbolic_decl,
      $.let_decl,
      $.var_decl,
      $.apply_stmt,
      $.measure_stmt,
      $.repeat_stmt,
      $.repeat_range_stmt,
      $.when_stmt,
      $.oracle_stmt,
    ),

    qreg_decl: $ => seq('qreg', field('name', $.identifier), '[', field('size', $._expression), ']'),
    creg_decl: $ => seq('creg', field('name', $.identifier), '[', field('size', $._expression), ']'),
    symbolic_decl: $ => seq('symbolic', field('name', $.identifier), '[', field('count', $._expression), ']'),

    let_decl: $ => seq('let', field('name', $.identifier), '=', field('value', $._expression)),
    var_decl: $ => seq('var', field('name', $.identifier), '=', field('value', $._expression)),

    apply_stmt: $ => seq(
      'apply',
      field('gate', $.gate_name),
      optional(field('params', $.gate_params)),
      'on',
      field('qubits', $.qubit_list),
    ),

    gate_name: _ => choice(
      'I', 'ID', 'X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG', 'SX',
      'RX', 'RY', 'RZ', 'P', 'U1', 'U', 'U3',
      'CX', 'CNOT', 'CY', 'CZ', 'SWAP', 'CP',
      'CCX', 'TOFFOLI', 'CSWAP', 'FREDKIN',
      'RXX', 'RYY', 'RZZ',
    ),

    gate_params: $ => seq(
      '(',
      optional(seq($._expression, repeat(seq(',', $._expression)))),
      ')',
    ),

    qubit_list: $ => seq($._target, repeat(seq(',', $._target))),

    _target: $ => choice($.indexed_register, $.identifier),

    indexed_register: $ => seq(
      field('name', $.identifier),
      '[',
      field('index', $._expression),
      ']',
    ),

    measure_stmt: $ => seq(
      'measure',
      field('from', $._target),
      '->',
      field('to', $._target),
    ),

    repeat_stmt: $ => seq(
      'repeat',
      field('count', $._expression),
      field('body', $.block),
    ),

    repeat_range_stmt: $ => seq(
      'repeat',
      field('var', $.identifier),
      'from', field('from', $._expression),
      'to',   field('to',   $._expression),
      optional(seq('step', field('step', $._expression))),
      field('body', $.block),
    ),

    when_stmt: $ => seq(
      'when',
      field('condition', $._expression),
      field('body', $.block),
    ),

    oracle_stmt: $ => seq(
      'oracle',
      field('name', $.identifier),
      optional(field('args', $.gate_params)),
      'on',
      field('qubits', $.qubit_list),
    ),

    // --- Expressions --------------------------------------------------------
    _expression: $ => choice(
      $.binary_expr,
      $.unary_expr,
      $.call_expr,
      $.indexed_register,
      $.identifier,
      $.integer,
      $.float,
      $.pi,
      $.boolean,
      seq('(', $._expression, ')'),
    ),

    binary_expr: $ => choice(
      prec.left(PREC.compare, seq($._expression, choice('==', '!=', '<', '<=', '>', '>='), $._expression)),
      prec.left(PREC.sum,     seq($._expression, choice('+', '-'),                       $._expression)),
      prec.left(PREC.product, seq($._expression, choice('*', '/', '%'),                  $._expression)),
      prec.right(PREC.power,  seq($._expression, '^',                                    $._expression)),
    ),

    unary_expr: $ => prec(PREC.unary, seq('-', $._expression)),

    call_expr: $ => prec(PREC.call, seq(
      field('callee', $.identifier),
      '(',
      optional(seq($._expression, repeat(seq(',', $._expression)))),
      ')',
    )),

    // --- Lexical primitives -------------------------------------------------
    identifier: _ => /[a-zA-Z_][a-zA-Z0-9_]*/,
    type_identifier: $ => alias($.identifier, $.type_identifier),

    integer: _ => /[0-9]+/,
    float:   _ => /[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?/,

    pi: _ => 'pi',
    boolean: _ => choice('true', 'false'),

    string: _ => seq('"', /[^"]*/, '"'),

    // Bra/ket markers used inside @prove blocks.
    ket: _ => /\|[+\-0-9a-zA-Z_]*>/,
    bra: _ => /<[+\-0-9a-zA-Z_]*\|/,
  },
});
